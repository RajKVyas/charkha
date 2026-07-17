"""CHARKHA unified CLI — single entry point for all commands."""

import argparse
import sys
import subprocess


def main():
    parser = argparse.ArgumentParser(
        description="CHARKHA — depth-recurrent hybrid language model for consumer GPUs",
        usage="charkha <command> [<args>]",
    )
    parser.add_argument(
        "command",
        choices=["toy", "train", "serve", "preflight", "pipeline", "test"],
        help="Command to run",
    )
    parser.add_argument("args", nargs=argparse.REMAINDER, help="Arguments for the command")
    args = parser.parse_args()

    if args.command == "test":
        sys.exit(subprocess.call([sys.executable, "-m", "pytest", "tests/", "-q"] + args.args))
    elif args.command == "toy":
        from charkha._toy import main as toy_main

        sys.argv = ["charkha-toy"] + args.args
        toy_main()
    elif args.command == "train":
        from train_cli import main as train_main

        sys.argv = ["charkha-train"] + args.args
        train_main()
    elif args.command == "serve":
        from serve import main as serve_main

        sys.argv = ["charkha-serve"] + args.args
        serve_main()
    elif args.command == "preflight":
        from preflight import main as pf_main

        sys.argv = ["charkha-preflight"] + args.args
        pf_main()
    elif args.command == "pipeline":
        from pipeline_cli import main as pl_main

        sys.argv = ["charkha-pipeline"] + args.args
        pl_main()


if __name__ == "__main__":
    main()
