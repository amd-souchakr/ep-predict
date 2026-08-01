from ep_predict.cli import main


if __name__ == "__main__":
    raise SystemExit(main(["collect", *__import__("sys").argv[1:]]))
