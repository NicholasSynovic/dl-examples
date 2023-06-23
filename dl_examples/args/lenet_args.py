from argparse import ArgumentParser, Namespace
from importlib.metadata import version
from pathlib import Path

from dl_examples import args as argVars

PROGRAM_NAME: str = "Example LeNet CV Model"


def getArgs() -> Namespace:
    parser: ArgumentParser = ArgumentParser(
        prog=PROGRAM_NAME,
        description="An example LeNet model written using Jax and Flax",
        epilog=f"Created by: {', '.join(argVars.authorsList)}",
        formatter_class=argVars.AlphabeticalOrderHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{PROGRAM_NAME}: {version(distribution_name='dl-examples')}",
    )
    parser.add_argument(
        "-d",
        "--dataset",
        nargs=1,
        type=Path,
        required=True,
        help="Path to store/access MNIST dataset",
    )

    return parser.parse_args()
