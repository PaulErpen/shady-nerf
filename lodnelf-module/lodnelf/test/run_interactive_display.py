import argparse
from lodnelf.test.interactive_display import InteractiveDisplay


def main():
    args = argparse.ArgumentParser(description="Training")

    args.add_argument(
        "--config",
        type=str,
        required=True,
        help="Name of the config to use for training.",
    )

    args.add_argument(
        "--model_save_path",
        type=str,
        required=True,
        help="Path where the trained model is stored.",
    )

    args = args.parse_args()

    interactive_display = InteractiveDisplay(args.config, args.model_save_path)
    interactive_display.run()


if __name__ == "__main__":
    main()
