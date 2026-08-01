from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable


DISPOSABLE_DIRECTORY_NAMES = {
    "trace",
    "features",
    "hidden_states",
    "activations",
}
DEFAULT_LARGE_FILE_BYTES = 50 * 1024 * 1024
RESULT_REFERENCE_FILES = (
    "README.md",
    "STATUS.md",
    "EXPERIMENT_LOG.md",
)
ARTIFACT_REFERENCE_PATTERN = re.compile(r"artifacts/[A-Za-z0-9_./-]+")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_disposable(path: Path, artifacts_root: Path) -> bool:
    try:
        relative = path.relative_to(artifacts_root)
    except ValueError:
        return False
    return any(part in DISPOSABLE_DIRECTORY_NAMES for part in relative.parts)


def _manifest_entries(section: Any) -> Iterable[tuple[str, str]]:
    if isinstance(section, list):
        for item in section:
            if isinstance(item, dict) and "path" in item and "sha256" in item:
                yield str(item["path"]), str(item["sha256"])
        return

    if not isinstance(section, dict):
        return

    for key, value in section.items():
        if isinstance(value, dict) and "path" in value and "sha256" in value:
            yield str(value["path"]), str(value["sha256"])
        elif isinstance(value, str):
            yield str(key), value


def _git_paths(repo_root: Path, *arguments: str) -> set[str]:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
    )
    if completed.returncode not in {0, 1}:
        raise RuntimeError(completed.stderr.strip() or "git command failed")
    return {line for line in completed.stdout.splitlines() if line}


def _document_reference_files(repo_root: Path) -> list[Path]:
    paths = [repo_root / name for name in RESULT_REFERENCE_FILES]
    paths.extend(sorted((repo_root / "docs").glob("*_RESULTS.md")))
    return [path for path in paths if path.is_file()]


def audit_artifacts(
    repo_root: Path,
    *,
    require_tracked: bool = False,
    large_file_bytes: int = DEFAULT_LARGE_FILE_BYTES,
) -> dict[str, Any]:
    root = repo_root.resolve()
    artifacts_root = root / "artifacts"
    errors: list[str] = []
    warnings: list[str] = []

    if not artifacts_root.is_dir():
        return {
            "state": "failed",
            "errors": ["missing artifacts directory"],
            "warnings": [],
        }

    artifact_files = sorted(path for path in artifacts_root.rglob("*") if path.is_file())
    relative_files = [str(path.relative_to(root)) for path in artifact_files]
    ignored = _git_paths(root, "check-ignore", *relative_files)
    tracked = _git_paths(root, "ls-files", "artifacts")

    durable_count = 0
    durable_bytes = 0
    disposable_count = 0
    disposable_bytes = 0

    for path, relative in zip(artifact_files, relative_files, strict=True):
        disposable = _is_disposable(path, artifacts_root)
        if disposable:
            disposable_count += 1
            disposable_bytes += path.stat().st_size
            if relative not in ignored:
                errors.append(f"disposable raw artifact is not ignored: {relative}")
            if relative in tracked:
                errors.append(f"disposable raw artifact is tracked: {relative}")
            continue

        durable_count += 1
        size = path.stat().st_size
        durable_bytes += size
        if relative in ignored:
            errors.append(f"durable artifact is ignored: {relative}")
        if require_tracked and relative not in tracked:
            errors.append(f"durable artifact is not staged/tracked: {relative}")
        if size > large_file_bytes:
            warnings.append(
                f"large durable artifact ({size / 1024 / 1024:.1f} MiB): "
                f"{relative}"
            )

    checked_references = 0
    manifests = sorted(artifacts_root.rglob("figure_manifest.json"))
    for manifest_path in manifests:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"cannot read figure manifest {manifest_path}: {error}")
            continue
        for section_name in ("inputs", "outputs"):
            for relative, expected_hash in _manifest_entries(
                manifest.get(section_name)
            ):
                checked_references += 1
                target = root / relative
                if not target.is_file():
                    errors.append(
                        f"missing figure {section_name[:-1]} referenced by "
                        f"{manifest_path.relative_to(root)}: {relative}"
                    )
                    continue
                actual_hash = _sha256(target)
                if actual_hash != expected_hash:
                    errors.append(
                        f"hash mismatch for {relative}: expected "
                        f"{expected_hash}, found {actual_hash}"
                    )

    checked_document_references = 0
    for document in _document_reference_files(root):
        text = document.read_text(encoding="utf-8")
        for match in ARTIFACT_REFERENCE_PATTERN.finditer(text):
            relative = match.group().rstrip("./")
            checked_document_references += 1
            if not (root / relative).exists():
                errors.append(
                    f"stale artifact reference in {document.relative_to(root)}: "
                    f"{relative}"
                )

    return {
        "state": "complete" if not errors else "failed",
        "require_tracked": require_tracked,
        "durable": {
            "files": durable_count,
            "bytes": durable_bytes,
        },
        "disposable": {
            "files": disposable_count,
            "bytes": disposable_bytes,
        },
        "figure_manifests": len(manifests),
        "figure_references_checked": checked_references,
        "document_references_checked": checked_document_references,
        "errors": errors,
        "warnings": warnings,
    }
