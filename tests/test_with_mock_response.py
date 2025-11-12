#!/usr/bin/env python3
"""
Test the spec-kit-next-step.py script with a mock LLM response to verify complete workflow.
"""

import json
import sys
from pathlib import Path

# Add src directory to path so we can import the script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# Mock the requests.post method to return a successful response
class MockResponse:
    def __init__(self):
        self.status_code = 200
        self._data = {
            "choices": [
                {
                    "message": {
                        "content": "/speckit.implement"
                    }
                }
            ]
        }

    def json(self):
        return self._data

# Mock requests to avoid actual API calls
import unittest.mock

def test_with_mock_llm():
    """Test the complete workflow with a mock LLM response."""

    # Test data
    hook_input = {
        "transcript_path": str(Path(__file__).parent / "test_transcript.jsonl"),
        "session_id": "test-session-456",
        "event_type": "tool_call"
    }

    print("Testing spec-kit-next-step.py with mock LLM response...")

    # Change to test directory
    test_dir = Path(__file__).parent
    import os
    original_cwd = os.getcwd()
    os.chdir(test_dir)

    try:
        # Mock the requests.post method
        with unittest.mock.patch('requests.post', return_value=MockResponse()):
            # Mock subprocess.Popen to avoid actually calling claude
            with unittest.mock.patch('subprocess.Popen') as mock_popen:
                # Configure mock process
                mock_process = unittest.mock.MagicMock()
                mock_process.communicate.return_value = ("mock output", "")
                mock_popen.return_value = mock_process

                # Import and run the main function
                from spec_kit_next_step import main

                # Mock stdin to provide our test input
                import io
                mock_stdin = io.StringIO(json.dumps(hook_input))
                sys.stdin = mock_stdin

                try:
                    main()
                    print("✅ Script completed successfully with mock LLM response")

                    # Check the log file
                    log_file = test_dir / "log" / "hook_content.log"
                    if log_file.exists():
                        with open(log_file, 'r') as f:
                            log_content = f.read()
                            if "/speckit.implement" in log_content:
                                print("✅ Mock LLM response was processed correctly")
                                if "test-session-456" in log_content:
                                    print("✅ Session ID was handled correctly")
                                    return True
                                else:
                                    print("❌ Session ID not found in log")
                                    return False
                            else:
                                print("❌ Mock LLM response not found in log")
                                return False
                    else:
                        print("❌ Log file not created")
                        return False

                except SystemExit as e:
                    print(f"❌ Script exited with code: {e.code}")
                    return False
                except Exception as e:
                    print(f"❌ Script failed with error: {e}")
                    return False
                finally:
                    sys.stdin = sys.__stdin__

    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":
    success = test_with_mock_llm()
    sys.exit(0 if success else 1)