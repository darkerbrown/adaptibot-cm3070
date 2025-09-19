"""Test setup helper used by pytest to ensure the project root is importable.

This file is only used during automated tests to make sure the `ant_rescue`
package can be imported when tests run from the repository root.
"""

import sys
from pathlib import Path
root = Path(__file__).resolve().parent.parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
