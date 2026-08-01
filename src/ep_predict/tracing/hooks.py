from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from typing import Any

import torch

from ep_predict.tracing.schema import RequestContext, TraceRecord


_LAYER_PATTERN = re.compile(r"(?:layers|blocks|h)\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class RouterSpec:
    name: str
    layer_id: int
    moe_layer_index: int
    num_experts: int
    top_k: int
    module: torch.nn.Module


class RouterInputProjector:
    """Fixed random-sign projection applied to the tensor entering a router."""

    def __init__(
        self,
        *,
        input_dimension: int,
        output_dimension: int,
        seed: int,
        storage_dtype: str = "float16",
    ) -> None:
        if input_dimension <= 0 or output_dimension <= 0:
            raise ValueError("projection dimensions must be positive")
        if storage_dtype != "float16":
            raise ValueError("the H3 prototype supports float16 feature storage")
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        bits = torch.randint(
            0,
            2,
            (input_dimension, output_dimension),
            dtype=torch.uint8,
            generator=generator,
        )
        self.input_dimension = input_dimension
        self.output_dimension = output_dimension
        self.seed = seed
        self.storage_dtype = storage_dtype
        self.scale = 1.0 / math.sqrt(output_dimension)
        self._signs = bits.to(torch.int8).mul_(2).sub_(1)
        self._matrix_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.matrix_sha256 = hashlib.sha256(memoryview(bits.numpy())).hexdigest()

    def project(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if hidden_states.shape[-1] != self.input_dimension:
            raise ValueError(
                f"router input dimension {hidden_states.shape[-1]} does not "
                f"match projection input {self.input_dimension}"
            )
        key = (str(hidden_states.device), hidden_states.dtype)
        matrix = self._matrix_cache.get(key)
        if matrix is None:
            matrix = self._signs.to(
                device=hidden_states.device,
                dtype=hidden_states.dtype,
            )
            matrix.mul_(self.scale)
            self._matrix_cache[key] = matrix
        projected = hidden_states.reshape(-1, self.input_dimension) @ matrix
        return projected.detach().to(device="cpu", dtype=torch.float16)

    def report(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "point": "router_input_positional_argument_0",
            "input_dimension": self.input_dimension,
            "output_dimension": self.output_dimension,
            "projection_family": "rademacher",
            "projection_seed": self.seed,
            "projection_scale": self.scale,
            "matrix_sha256": self.matrix_sha256,
            "storage_dtype": self.storage_dtype,
        }


def _integer_attribute(module: torch.nn.Module, *names: str) -> int | None:
    for name in names:
        value = getattr(module, name, None)
        if isinstance(value, int) and value > 0:
            return value
    return None


def discover_routers(
    model: torch.nn.Module,
    name_contains: list[str] | None = None,
) -> list[RouterSpec]:
    """Find explicit router modules without importing model-specific classes."""
    candidates: list[tuple[str, int, int, int, torch.nn.Module]] = []
    seen_names: set[str] = set()
    for name, module in model.named_modules():
        num_experts = _integer_attribute(module, "num_experts", "n_experts")
        top_k = _integer_attribute(
            module, "top_k", "num_experts_per_tok", "num_selected_experts"
        )
        if num_experts is None or top_k is None:
            continue

        class_name = module.__class__.__name__.lower()
        looks_like_router = (
            "router" in class_name
            or "router" in name.lower()
            or name.endswith(".gate")
        )
        if name_contains:
            looks_like_router = looks_like_router and any(
                fragment in name for fragment in name_contains
            )
        if not looks_like_router:
            continue

        layer_match = _LAYER_PATTERN.search(name)
        layer_id = int(layer_match.group(1)) if layer_match else len(candidates)
        candidates.append((name, layer_id, num_experts, top_k, module))
        seen_names.add(name)

    # Older/fused implementations may keep expert counts on the MoE block and
    # expose the gate itself as a plain Linear. The hook point remains explicit.
    for parent_name, parent in model.named_modules():
        num_experts = _integer_attribute(parent, "num_experts", "n_experts")
        top_k = _integer_attribute(
            parent, "top_k", "num_experts_per_tok", "num_selected_experts"
        )
        gate = getattr(parent, "gate", None)
        if (
            num_experts is None
            or top_k is None
            or not isinstance(gate, torch.nn.Module)
        ):
            continue
        name = f"{parent_name}.gate" if parent_name else "gate"
        if name in seen_names:
            continue
        if name_contains and not any(fragment in name for fragment in name_contains):
            continue
        layer_match = _LAYER_PATTERN.search(name)
        layer_id = int(layer_match.group(1)) if layer_match else len(candidates)
        candidates.append((name, layer_id, num_experts, top_k, gate))
        seen_names.add(name)

    candidates.sort(key=lambda candidate: (candidate[1], candidate[0]))
    return [
        RouterSpec(
            name=name,
            layer_id=layer_id,
            moe_layer_index=index,
            num_experts=num_experts,
            top_k=top_k,
            module=module,
        )
        for index, (name, layer_id, num_experts, top_k, module) in enumerate(candidates)
    ]


def _extract_router_output(
    output: Any,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
    """Return logits, selected weights, and actual selected IDs."""
    if isinstance(output, (tuple, list)) and len(output) >= 3:
        logits, weights, selected = output[:3]
        if (
            isinstance(logits, torch.Tensor)
            and isinstance(selected, torch.Tensor)
            and selected.dtype
            in {
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            }
        ):
            return logits, weights if isinstance(weights, torch.Tensor) else None, selected

    if isinstance(output, torch.Tensor):
        logits = output
        probabilities = torch.softmax(logits.float(), dim=-1)
        weights, selected = torch.topk(probabilities, top_k, dim=-1)
        return logits, weights, selected

    raise TypeError(
        "router output must be logits or a (logits, weights, selected_ids) tuple"
    )


class RouterTracer:
    """Capture router decisions with hooks while leaving model code untouched."""

    def __init__(
        self,
        model: torch.nn.Module,
        routers: list[RouterSpec],
        *,
        fail_on_router_mismatch: bool = True,
        fail_on_missing_router: bool = True,
        feature_projector: RouterInputProjector | None = None,
    ) -> None:
        if not routers:
            raise ValueError("no router modules discovered")
        self.model = model
        self.routers = routers
        self.fail_on_router_mismatch = fail_on_router_mismatch
        self.fail_on_missing_router = fail_on_missing_router
        self.feature_projector = feature_projector
        self.records: list[TraceRecord] = []
        self.hidden_features: list[torch.Tensor] = []
        self.context: RequestContext | None = None
        self._request_forward_count = 0
        self._batch_id = -1
        self._active_rows: list[tuple[int, int, int]] = []
        self._active_batch_size = 0
        self._active_phase = "prefill"
        self._routers_seen: set[str] = set()
        self._call_router_counts: list[int] = []
        self.router_validation_mismatches = 0
        self._handles: list[Any] = []

    def __enter__(self) -> RouterTracer:
        self._handles.append(
            self.model.register_forward_pre_hook(self._root_pre_hook, with_kwargs=True)
        )
        self._handles.append(
            self.model.register_forward_hook(self._root_post_hook, with_kwargs=True)
        )
        for spec in self.routers:
            self._handles.append(
                spec.module.register_forward_hook(self._make_router_hook(spec))
            )
        return self

    def __exit__(self, *_: object) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def start_request(self, context: RequestContext) -> None:
        if self.context is not None:
            raise RuntimeError("finish the active request before starting another")
        self.context = context
        self.records = []
        self.hidden_features = []
        self._request_forward_count = 0
        self._call_router_counts = []
        self.router_validation_mismatches = 0

    def finish_request(
        self,
    ) -> tuple[list[TraceRecord], torch.Tensor | None, dict[str, Any]]:
        if self.context is None:
            raise RuntimeError("no active request")
        missing_calls = [
            index
            for index, count in enumerate(self._call_router_counts)
            if count != len(self.routers)
        ]
        if missing_calls and self.fail_on_missing_router:
            raise RuntimeError(
                f"expected {len(self.routers)} routers per forward; "
                f"bad calls: {missing_calls[:8]}"
            )
        if self.router_validation_mismatches and self.fail_on_router_mismatch:
            raise RuntimeError(
                f"{self.router_validation_mismatches} router selections did not "
                "match top-k logits"
            )
        records = self.records
        features = (
            torch.stack(self.hidden_features)
            if self.feature_projector is not None
            else None
        )
        if features is not None and len(features) != len(records):
            raise RuntimeError(
                f"captured {len(features)} hidden features for {len(records)} "
                "routing records"
            )
        summary = {
            "records": len(records),
            "hidden_feature_records": len(self.hidden_features),
            "model_forward_calls": self._request_forward_count,
            "router_count": len(self.routers),
            "router_validation_mismatches": self.router_validation_mismatches,
            "router_calls_per_forward": self._call_router_counts,
        }
        self.context = None
        self.records = []
        self.hidden_features = []
        return records, features, summary

    def _root_pre_hook(
        self,
        _module: torch.nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]] | None:
        if self.context is None:
            return None
        input_ids = kwargs.get("input_ids")
        if input_ids is None and args:
            input_ids = args[0]
        if not isinstance(input_ids, torch.Tensor) or input_ids.ndim != 2:
            raise ValueError("hook tracing requires rank-2 input_ids")
        if input_ids.shape[0] != 1:
            raise ValueError(
                "the H1 prototype intentionally uses batch size 1; "
                "batch tracing will be added when wave experiments begin"
            )

        attention_mask = kwargs.get("attention_mask")
        if not isinstance(attention_mask, torch.Tensor):
            attention_mask = torch.ones_like(input_ids)

        ids = input_ids.detach().to("cpu")
        mask_tail = attention_mask[:, -input_ids.shape[1] :].detach().to("cpu")
        full_mask = attention_mask.detach().to("cpu")
        self._active_phase = (
            "prefill" if self._request_forward_count == 0 else "decode"
        )
        self._active_batch_size = int(input_ids.shape[0])
        self._active_rows = []

        if self._active_phase == "prefill":
            positions = full_mask.cumsum(dim=1)[:, -input_ids.shape[1] :] - 1
            for column in range(input_ids.shape[1]):
                if int(mask_tail[0, column]) == 0:
                    continue
                self._active_rows.append(
                    (column, int(positions[0, column]), int(ids[0, column]))
                )
        else:
            valid_columns = torch.nonzero(mask_tail[0], as_tuple=False).flatten()
            if len(valid_columns) == 0:
                raise ValueError("decode call contains no unmasked token")
            column = int(valid_columns[-1])
            flat_row = column
            token_position = int(full_mask[0].sum().item()) - 1
            self._active_rows.append((flat_row, token_position, int(ids[0, column])))

        self._batch_id += 1
        self._request_forward_count += 1
        self._routers_seen = set()
        return None

    def _root_post_hook(
        self,
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        _kwargs: dict[str, Any],
        output: Any,
    ) -> Any:
        if self.context is not None:
            self._call_router_counts.append(len(self._routers_seen))
        return output

    def _make_router_hook(self, spec: RouterSpec):
        def hook(
            _module: torch.nn.Module,
            _inputs: tuple[Any, ...],
            output: Any,
        ) -> None:
            if self.context is None:
                return
            logits, weights, selected = _extract_router_output(output, spec.top_k)
            projected_cpu: torch.Tensor | None = None
            if self.feature_projector is not None:
                if not _inputs or not isinstance(_inputs[0], torch.Tensor):
                    raise TypeError(
                        f"router {spec.name} did not expose a tensor as "
                        "positional input zero"
                    )
                projected_cpu = self.feature_projector.project(_inputs[0])
            logits_2d = logits.reshape(-1, logits.shape[-1])
            selected_2d = selected.reshape(-1, selected.shape[-1])
            weights_2d = (
                weights.reshape(-1, weights.shape[-1]) if weights is not None else None
            )
            expected = torch.topk(logits_2d.float(), spec.top_k, dim=-1).indices
            mismatch_count = int((expected != selected_2d).any(dim=-1).sum().item())
            self.router_validation_mismatches += mismatch_count

            selected_cpu = selected_2d.detach().to("cpu")
            weights_cpu = (
                weights_2d.detach().float().to("cpu")
                if weights_2d is not None
                else None
            )
            for flat_row, token_position, token_id in self._active_rows:
                if flat_row >= selected_cpu.shape[0]:
                    raise ValueError(
                        f"router {spec.name} emitted {selected_cpu.shape[0]} rows, "
                        f"but trace requested row {flat_row}"
                    )
                expert_ids = [int(value) for value in selected_cpu[flat_row].tolist()]
                expert_weights = (
                    [float(value) for value in weights_cpu[flat_row].tolist()]
                    if weights_cpu is not None
                    else []
                )
                self.records.append(
                    TraceRecord(
                        run_id=self.context.run_id,
                        request_id=self.context.request_id,
                        sample_id=self.context.sample_id,
                        phase=self._active_phase,
                        token_position=token_position,
                        input_token_id=token_id,
                        layer_id=spec.layer_id,
                        moe_layer_index=spec.moe_layer_index,
                        selected_expert_ids=expert_ids,
                        selected_expert_weights=expert_weights,
                        batch_id=self._batch_id,
                        batch_size=self._active_batch_size,
                        dataset_name=self.context.dataset_name,
                        domain=self.context.domain,
                    )
                )
                if projected_cpu is not None:
                    if flat_row >= projected_cpu.shape[0]:
                        raise ValueError(
                            f"router {spec.name} projected "
                            f"{projected_cpu.shape[0]} rows, but trace requested "
                            f"row {flat_row}"
                        )
                    self.hidden_features.append(projected_cpu[flat_row].clone())
            self._routers_seen.add(spec.name)

        return hook
