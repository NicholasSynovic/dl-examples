from argparse import ArgumentDefaultsHelpFormatter
from typing import List

authorsList: List[str] = [
    "Nicholas M. Synovic",
]


class AlphabeticalOrderHelpFormatter(ArgumentDefaultsHelpFormatter):
    def add_arguments(self, actions):
        actions = sorted(actions, key=lambda x: x.dest)
        super(AlphabeticalOrderHelpFormatter, self).add_arguments(actions)
