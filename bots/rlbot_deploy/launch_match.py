import subprocess
import time
import os
import sys

# Path to your match.toml
MATCH_FILE = "match.toml"

def launch_rlbot_match():
    print("Starting RLBot match...")
    subprocess.run(["rlbot", "run", MATCH_FILE])

def main():
    if not os.path.exists(MATCH_FILE):
        print(f"Error: {MATCH_FILE} not found.")
        sys.exit(1)

    # Launch RLBot match
    launch_rlbot_match()


if __name__ == "__main__":
    main()
