#!/usr/bin/env python3
"""
Test script for extract_last_user_prompt_from_jsonl_file function from spec-kit-next-step.py
"""

import sys
import os
import json
import importlib.util
from pathlib import Path

# Import the spec-kit-next-step.py module
spec_kit_path = Path(__file__).parent.parent / 'src' / 'spec-kit-next-step.py'
spec_kit_spec = importlib.util.spec_from_file_location("spec_kit_module", spec_kit_path)
spec_kit_module = importlib.util.module_from_spec(spec_kit_spec)
spec_kit_spec.loader.exec_module(spec_kit_module)

extract_last_user_prompt_from_jsonl_file = spec_kit_module.extract_last_user_prompt_from_jsonl_file
should_process_hook = spec_kit_module.should_process_hook

def test_extract_last_user_prompt():
    """Test the extract_last_user_prompt_from_jsonl_file function."""

    print("=" * 80)
    print("TESTING extract_last_user_prompt_from_jsonl_file FUNCTION")
    print("=" * 80)
    print()

    # Test with real JSONL data file
    test_file = Path(__file__).parent / 'test_extract_last_user_prompt_negative.jsonl'

    print(f"Test file: {test_file}")
    print(f"File exists: {test_file.exists()}")
    print()

    if not test_file.exists():
        print("❌ Error: Test file not found!")
        return

    # Call the independent function directly
    print("🔍 Calling extract_last_user_prompt_from_jsonl_file...")
    print()

    last_user_prompt, line_number = extract_last_user_prompt_from_jsonl_file(test_file)

    # Display results
    if last_user_prompt:
        print("✅ SUCCESS: Function returned valid result")
        print(f"   Line number: {line_number}")
        print(f"   User prompt: {last_user_prompt[:200]}{'...' if len(last_user_prompt) > 200 else ''}")
        print()

        # Test filtering logic using the same logic as should_process_hook
        should_process, reason = should_process_hook(last_user_prompt)

        # Extract actual command for display
        actual_command = last_user_prompt.strip()
        if '<command-name>' in last_user_prompt:
            import re
            cmd_match = re.search(r'<command-name>(.*?)</command-name>', last_user_prompt)
            if cmd_match:
                actual_command = cmd_match.group(1).strip()

        print(f"   Actual command: '{actual_command}'")
        is_speckit = actual_command.startswith('/speckit.')
        is_constitution = actual_command.startswith('/speckit.constitution')
        should_process = is_speckit and not is_constitution

        print("🔍 Filtering Analysis:")
        print(f"   Should process: {should_process}")
        print(f"   Reason: {reason}")

        if should_process:
            print("✅ Hook would be PROCESSED (valid speckit command)")
        else:
            print("❌ Hook would be SKIPPED")
    else:
        print("❌ FAILED: Function returned None")
        print(f"   Line number: {line_number}")
        print("   No user prompt found in the JSONL file")

if __name__ == "__main__":
    test_extract_last_user_prompt()
