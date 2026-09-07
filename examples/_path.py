"""Put the project root on sys.path so the examples run from a checkout.

Only needed when the package has not been installed (`pip install -e .`).
Importing this module first keeps every example runnable either way.
"""

import sys
from pathlib import Path

root = str(Path(__file__).resolve().parent.parent)
if root not in sys.path:
    sys.path.insert(0, root)
