"""
Main entry point for the Planuze LLM CLI.

This module provides the Command Line Interface (CLI) for the Planuze LLM
application, allowing users to execute various commands such as data
preparation, model training, and running the full pipeline. It handles
argument parsing and dispatches commands to the appropriate use cases via
the dependency injection container.
"""

import argparse
import sys
from typing import List

from .container import Container


class CLI:
    """
    Command Line Interface for the Planuze LLM application.

    This class handles the initialization of the application container,
    manages command-line argument parsing, and dispatches requests to
    the appropriate use cases.
    """

    def __init__(self):
        """Initialize the CLI with the dependency injection container."""
        self.container = Container()
        self.presenter = self.container.presenter

    def _run_publish(self):
        """Execute the model publishing use case."""
        self.presenter.show_panel(
            "Publication",
            {
                "Repo": self.container.config.model.hf_repo_id,
            },
        )
        use_case = self.container.get_publish_model_use_case()
        use_case.execute(self.container.config)

    def run(self, args: List[str]):
        """Parse arguments and execute the requested command."""
        parser = argparse.ArgumentParser(description="Planuze LLM CLI")
        subparsers = parser.add_subparsers(dest="command", help="Command to execute")

        # Command: prepare
        prepare_parser = subparsers.add_parser("prepare", help="Prepare data")
        prepare_parser.add_argument(
            "--val-ratio", type=float, default=0.1, help="Validation validation"
        )

        # Command: train
        subparsers.add_parser("train", help="Train model")

        # Command: full
        subparsers.add_parser("full", help="Full pipeline (Prepare + Train)")

        # Command: publish
        subparsers.add_parser("publish", help="Publish model to Hugging Face")

        parsed_args = parser.parse_args(args)

        if not parsed_args.command:
            parser.print_help()
            return

        try:
            if parsed_args.command == "prepare":
                self._run_prepare(parsed_args)
            elif parsed_args.command == "train":
                self._run_train()
            elif parsed_args.command == "full":
                self._run_prepare(parsed_args)
                self._run_train()
            elif parsed_args.command == "publish":
                self._run_publish()

        except Exception as e:
            self.presenter.log(f"Fatal error: {str(e)}", "error")
            sys.exit(1)

    def _run_prepare(self, args):
        """Execute the data preparation use case."""
        self.presenter.show_panel("Data Preparation", {"Status": "Starting..."})
        use_case = self.container.get_prepare_data_use_case()

        stats = use_case.execute(
            self.container.config.data,
            val_ratio=args.val_ratio if hasattr(args, "val_ratio") else 0.1,
        )

        self.presenter.show_panel("Data Prepared", stats, style="green")

    def _run_train(self):
        """Execute the model training use case."""
        self.presenter.show_panel(
            "Training",
            {
                "Model": self.container.config.model.name,
                "Backend": self.container.config.backend.type.upper(),
            },
        )
        use_case = self.container.get_train_model_use_case()

        result = use_case.execute(self.container.config)

        self.presenter.show_panel("Training Finished", result, style="green")


def main():
    """Entry point for the CLI application."""
    cli = CLI()
    cli.run(sys.argv[1:])


if __name__ == "__main__":
    main()
