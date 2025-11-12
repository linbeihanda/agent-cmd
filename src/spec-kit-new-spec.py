#!/usr/bin/env python3
"""
Auto-run Script for Claude Code

This script reads requirement and constitution files, then opens a Claude Code session
with the /specify.specify command and requirement file content.
"""

import argparse
import sys
import os
import subprocess
import tempfile
from pathlib import Path


def read_file_content(file_path):
    """Read content from file and return as string."""
    try:
        path = Path(file_path)
        if not path.exists():
            print(f"Error: File '{file_path}' does not exist.", file=sys.stderr)
            sys.exit(1)

        with open(path, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}", file=sys.stderr)
        sys.exit(1)


def build_claude_command(requirements_content, constitution_content=None):
    """Build the Claude Code command with requirements and optional constitution."""
    # Start with the /specify.specify command
    command_parts = ["/speckit.specify"]

    # Add requirements content
    if requirements_content:
        command_parts.append(requirements_content)

    # Add constitution content if provided
    if constitution_content:
        command_parts.append(constitution_content)

    return " ".join(command_parts)


def run_claude_code_persistently(command):
    """Execute Claude Code using Popen for cross-platform persistent execution."""
    try:
        # Use Popen to start Claude Code with the command as input
        # This creates a detached process that continues running after our script exits
        process = subprocess.Popen(
            ['claude-code'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Detach from parent process so it continues running after script exit
            start_new_session=True if sys.platform != "win32" else 0
        )

        # Send the command to Claude Code's stdin
        stdout, stderr = process.communicate(input=command, timeout=5)

        print("Claude Code session started successfully.")
        print("The session is now running independently in the background.")
        print("You can interact with Claude Code through its interface.")

        # Note: The actual Claude Code GUI/interface will open and continue running
        # even though this script exits. The Popen process is detached.

    except subprocess.TimeoutExpired:
        # This is actually expected - Claude Code may keep running
        print("Claude Code session started successfully.")
        print("The session is running in the background.")

    except FileNotFoundError:
        print("Error: 'claude-code' command not found.", file=sys.stderr)
        print("Please make sure Claude Code is installed and in your PATH.", file=sys.stderr)
        sys.exit(1)

    except Exception as e:
        print(f"Error launching Claude Code: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    """Main function to parse arguments and launch Claude Code."""
    parser = argparse.ArgumentParser(
        description="Auto-run Claude Code with /specify.specify command and requirement file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -r requirements.txt
  %(prog)s --req requirements.txt --consti constitution.txt
  %(prog)s -r user_story.md -c project_rules.md
        """
    )

    parser.add_argument(
        '-r', '--req', '--requirements',
        required=True,
        help='Path to the requirements file (user requirement)'
    )

    parser.add_argument(
        '-c', '--consti', '--constitution',
        help='Path to the constitution file (optional)'
    )

    args = parser.parse_args()

    # Read requirement file
    print(f"Reading requirements from: {args.req}")
    requirements_content = read_file_content(args.req)

    if not requirements_content:
        print("Error: Requirements file is empty.", file=sys.stderr)
        sys.exit(1)

    # Read constitution file if provided
    constitution_content = None
    if args.consti:
        print(f"Reading constitution from: {args.consti}")
        constitution_content = read_file_content(args.consti)

    # Build the Claude Code command
    command = build_claude_command(requirements_content, constitution_content)

    print(f"Built command: /speckit.specify [requirements" +
          (" + constitution]" if constitution_content else "]"))

    # Execute Claude Code persistently
    run_claude_code_persistently(command)


if __name__ == "__main__":
    main()