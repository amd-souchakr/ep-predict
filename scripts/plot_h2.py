from ep_predict.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["plot-h2", *__import__("sys").argv[1:]]))
