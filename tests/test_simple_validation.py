#!/usr/bin/env python3
"""
Simple validation test to ensure the script processes our test data correctly.
"""

import json
import subprocess
import sys
from pathlib import Path

def test_simple_validation():
    """Test that the script can process our test data without errors."""

    test_dir = Path(__file__).parent
    script_path = test_dir.parent / "src" / "spec-kit-next-step.py"

    # Create test input
    test_input = {
        "transcript_path": str(test_dir / "test_transcript.jsonl"),
        "session_id": "validation-test"
    }

    # Run the script
    result = subprocess.run(
        [sys.executable, str(script_path)],
        input=json.dumps(test_input),
        text=True,
        capture_output=True,
        cwd=str(test_dir),
        timeout=30
    )

    # Check results
    print(f"Return code: {result.returncode}")
    print(f"STDERR: {result.stderr}")

    # Check if log was created and contains expected content
    log_file = test_dir / "log" / "hook_content.log"
    if log_file.exists():
        with open(log_file, 'r') as f:
            log_content = f.read()

        expected_content = [
            "validation-test",
            "Perfect! I can see that significant progress has already been made",
            "Session validation-test trigger count:"
        ]

        all_found = True
        for content in expected_content:
            if content in log_content:
                print(f"✅ Found expected content: {content[:50]}...")
            else:
                print(f"❌ Missing expected content: {content}")
                all_found = False

        return all_found
    else:
        print("❌ Log file not created")
        return False

if __name__ == "__main__":
    success = test_simple_validation()
    print(f"\nTest {'PASSED' if success else 'FAILED'}")
    sys.exit(0 if success else 1)