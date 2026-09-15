"""Where the json artifacts live.

Everything the scripts read or write goes in data/, one level up from src/. Use
data("name.json") rather than a bare filename, so the scripts work from any
working directory.
"""

import os

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")


def data(name):
    os.makedirs(DATA, exist_ok=True)
    return os.path.normpath(os.path.join(DATA, name))
