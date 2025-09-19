"""Small launcher for the Ant Rescue demo.

- This script is the simple command used to start the Ant Rescue
    application when you just want to run one rollout. It calls the
    higher-level `adaptibot.app.main()` function which converts command
    line options into a `RunConfig` and starts the controller.

Usage:
- python adaptibot_main.py --model <path-to-model>
"""

from adaptibot.app import main

if __name__ == "__main__":
        main()
