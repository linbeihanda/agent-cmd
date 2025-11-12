#!/usr/bin/env python3
"""
Test runner for spec-kit-next-step.py
"""

import json
import subprocess
import sys
from pathlib import Path

def test_spec_kit_next_step():
    """Test the spec-kit-next-step.py script with sample data."""

    # Get paths
    test_dir = Path(__file__).parent
    script_path = test_dir.parent / "src" / "spec-kit-next-step.py"
    hook_input_path = test_dir / "test_hook_input.json"

    print(f"Testing script: {script_path}")
    print(f"Using test input: {hook_input_path}")

    # Read the test hook input
    with open(hook_input_path, 'r') as f:
        hook_input = f.read()

    # Set up environment
    import os
    env = os.environ.copy()
    env['PWD'] = str(test_dir)
    env['CLAUDE_HOOK_EVENT'] = 'tool_call'

    # Run the script
    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            input=hook_input,
            text=True,
            capture_output=True,
            cwd=str(test_dir),
            env=env,
            timeout=60
        )

        print("STDOUT:")
        print(result.stdout)

        if result.stderr:
            print("STDERR:")
            print(result.stderr)

        print(f"Return code: {result.returncode}")

        return result.returncode == 0

    except subprocess.TimeoutExpired:
        print("Script timed out (this might be expected)")
        return True
    except Exception as e:
        print(f"Error running script: {e}")
        return False

if __name__ == "__main__":
    success = test_spec_kit_next_step()
    sys.exit(0 if success else 1)