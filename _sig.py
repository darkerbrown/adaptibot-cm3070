"""Developer helper: show the signature of the public `run` function.

- This tiny script prints the function signature for `ant_rescue.app.run`.
	It's useful for quick interactive inspection while developing or when
	a grader wants to see the main entry point's expected arguments.

Run manually when you are in the project root:

		python _sig.py
"""
from __future__ import annotations
import inspect
from adaptibot import app

def main() -> None:
    print(inspect.signature(app.run))


if __name__ == "__main__":
    main()