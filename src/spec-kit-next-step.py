#!/usr/bin/env python3
"""
Claude Code Hook Script: next-step.py

This script reads hook content, extracts transcript_path, reads JSONL file,
and extracts Claude Code's latest response.
"""

import json
import sys
import os
import subprocess
import time
import requests
from datetime import datetime
from pathlib import Path

def print_log(log_entry):
    # Prepare log directory and file (relative to script location)
    script_dir = Path(__file__).parent.parent # business project folder
    log_dir = script_dir / "log"
    log_file = log_dir / "hook_content.log"

    # Append to log file
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(log_entry + '\n')

def extract_content_safely(content):
    """Extract text content from either string or list format."""
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        text_parts = []
        for item in content:
            if isinstance(item, dict):
                # Handle different content types
                if item.get('type') == 'text':
                    text_parts.append(item.get('text', ''))
                elif 'text' in item:
                    text_parts.append(item['text'])
        return '\n'.join(text_parts)
    return ''


def extract_session_id(json_data):
    """Extract session ID from JSON data."""
    if isinstance(json_data, dict):
        # Look for common session ID fields
        session_fields = ['session_id', 'sessionId', 'session', 'conversation_id', 'conversationId']
        for field in session_fields:
            if field in json_data and json_data[field]:
                return str(json_data[field])

        # Check in nested structures
        if 'message' in json_data and isinstance(json_data['message'], dict):
            message = json_data['message']
            for field in session_fields:
                if field in message and message[field]:
                    return str(message[field])

        # Check in metadata or info sections
        for section in ['metadata', 'info', 'context', 'config']:
            if section in json_data and isinstance(json_data[section], dict):
                section_data = json_data[section]
                for field in session_fields:
                    if field in section_data and section_data[field]:
                        return str(section_data[field])

    return None


def get_session_counter_path(session_id):
    """Get the counter file path for a given session ID."""
    script_dir = Path(__file__).parent.parent
    log_dir = script_dir / "log"
    session_counters_dir = log_dir / "session_counters"
    session_counters_dir.mkdir(exist_ok=True)
    return session_counters_dir / f"{session_id}.count"


def load_env_vars():
    """Load environment variables from .env file if it exists."""
    env_vars = {}
    # Look for .env in the project root directory (agent-cmd/../)
    script_dir = Path(__file__).parent.parent
    env_file = script_dir / '.env'

    if env_file.exists():
        with open(env_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()

    return env_vars


def load_llm_config():
    """Load LLM configuration from environment variables with PM_AGENT_ prefix."""
    env_vars = load_env_vars()

    config = {
        'provider': env_vars.get('PM_AGENT_PROVIDER', 'openai'),
        'api_key': env_vars.get('PM_AGENT_API_KEY', ''),
        'base_url': env_vars.get('PM_AGENT_BASE_URL', 'https://api.openai.com/v1'),
        'model': env_vars.get('PM_AGENT_MODEL', 'gpt-3.5-turbo'),
        'prompt_template': env_vars.get('PM_AGENT_PROMPT_TEMPLATE', 'speckit_command_suggestion.txt')
    }

    return config


def check_and_increment_session_counter(session_id):
    """Check if session trigger count is within limit and increment if allowed."""
    if not session_id:
        print_log("Warning: No session ID provided, skipping counter check")
        return False

    # Load max_triggers from environment variables
    env_vars = load_env_vars()
    max_triggers = int(env_vars.get('MAX_TRIGGERS', '3'))

    counter_path = get_session_counter_path(session_id)

    # Read current count
    if counter_path.exists():
        with open(counter_path, 'r', encoding='utf-8') as f:
            current_count = int(f.read().strip())
    else:
        current_count = 0

    # Check limit
    if current_count >= max_triggers:
        print_log(f"Session {session_id} has reached maximum trigger limit ({max_triggers})")
        return False

    # Increment and save count
    new_count = current_count + 1
    with open(counter_path, 'w', encoding='utf-8') as f:
        f.write(str(new_count))

    print_log(f"Session {session_id} trigger count: {new_count}/{max_triggers}")
    return True


def extract_claude_response(json_data):
    """Extract Claude Code's latest response from JSON data."""
    if isinstance(json_data, dict):
        # Look for common response fields
        response_fields = ['response', 'content', 'message', 'text', 'output', 'answer']
        for field in response_fields:
            if field in json_data and json_data[field]:
                content = extract_content_safely(json_data[field])
                if content:
                    return content

        # Check for messages array (common in chat formats)
        if 'messages' in json_data:
            messages = json_data['messages']
            if messages and isinstance(messages, list):
                # Look for last assistant message
                for msg in reversed(messages):
                    if isinstance(msg, dict) and msg.get('role') == 'assistant':
                        content = msg.get('content', '')
                        if content:
                            return extract_content_safely(content)

        # Check for conversation history
        if 'conversation' in json_data:
            conv = json_data['conversation']
            if isinstance(conv, list):
                for entry in reversed(conv):
                    if isinstance(entry, dict):
                        if entry.get('role') == 'assistant' or entry.get('type') == 'assistant':
                            content = entry.get('content', entry.get('text', ''))
                            if content:
                                return extract_content_safely(content)

        # Check for message object (common in transcript formats)
        if 'message' in json_data:
            message = json_data['message']
            if isinstance(message, dict) and message.get('role') == 'assistant':
                content = message.get('content', '')
                if content:
                    return extract_content_safely(content)

    return None


def construct_prompt_from_template(claude_response, template_name):
    """Construct prompt using template file."""
    script_dir = Path(__file__).parent.parent
    template_path = script_dir / 'prompt_templates' / template_name
    if not template_path.exists():
        raise FileNotFoundError(f"Prompt template file not found: {template_path}")

    with open(template_path, 'r', encoding='utf-8') as f:
        template_content = f.read()

    # Replace placeholder with actual Claude response
    prompt = template_content.replace('{claude_response}', claude_response)
    return prompt


def send_prompt_to_llm(prompt, config):
    """Send prompt to LLM and get response."""
    try:
        # Currently both OpenAI and Zhipu AI use compatible API formats
        # We can use a unified function for all providers
        return send_to_llm_api(prompt, config)

    except Exception as e:
        print(f"Error sending prompt to LLM: {e}", file=sys.stderr)
        return None


def log_llm_request_response(prompt, config, response_data=None, error_info=None):
    """Log LLM request metadata, prompt, and response to hook_content.log."""
    script_dir = Path(__file__).parent.parent
    log_dir = script_dir / "log"
    log_file = log_dir / "hook_content.log"

    separator = "\n" + "="*80 + "\n"
    timestamp = datetime.now().isoformat()

    # Create request metadata
    request_metadata = {
        'provider': config['provider'],
        'base_url': config['base_url'],
        'model': config['model'],
        'temperature': 0.3,
        'max_tokens': 10240,
        'prompt_length': len(prompt),
        'api_call_timestamp': timestamp
    }

    log_entry = f"""
{separator}
Timestamp: {timestamp}
--- LLM API Request Metadata ---
{json.dumps(request_metadata, indent=2, ensure_ascii=False)}
--- End LLM API Request Metadata ---

--- LLM API Request Content ---
{prompt}
--- End LLM API Request Content ---
"""

    if response_data:
        # Successful response
        log_entry += f"""
--- LLM API Response ---
Status Code: 200
Response Data:
{json.dumps(response_data, indent=2, ensure_ascii=False)}
--- End LLM API Response ---
"""
    elif error_info:
        # Error response
        log_entry += f"""
--- LLM API Error Response ---
Error Details:
{json.dumps(error_info, indent=2, ensure_ascii=False)}
--- End LLM API Error Response ---
"""

    # Write to log file
    try:
        log_dir.mkdir(exist_ok=True)
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_entry)
    except Exception as e:
        print_log(f"Error writing LLM request/response to log: {e}")


def send_to_llm_api(prompt, config):
    """Send prompt to LLM API (unified function for OpenAI-compatible APIs)."""
    try:
        headers = {
            'Authorization': f'Bearer {config["api_key"]}',
            'Content-Type': 'application/json'
        }

        data = {
            'model': config['model'],
            'messages': [
                {'role': 'user', 'content': prompt}
            ],
            'temperature': 0.3,
            'max_tokens': 10240
        }

        # Log request before making API call
        print_log(f"Sending request to {config['provider']} API...")

        response = requests.post(
            f'{config["base_url"]}/chat/completions',
            headers=headers,
            json=data,
            timeout=30
        )

        if response.status_code == 200:
            result = response.json()

            # Log successful request and response
            log_llm_request_response(prompt, config, response_data=result)

            return result['choices'][0]['message']['content'].strip()
        else:
            # Log failed request with error details
            error_info = {
                'status_code': response.status_code,
                'error_message': response.text,
                'headers': dict(response.headers),
                'url': f'{config["base_url"]}/chat/completions'
            }

            log_llm_request_response(prompt, config, error_info=error_info)

            provider_name = config['provider'].capitalize()
            print(f"Error: {provider_name} API request failed with status {response.status_code}: {response.text}", file=sys.stderr)
            return None

    except Exception as e:
        # Log exception error
        error_info = {
            'exception_type': type(e).__name__,
            'error_message': str(e),
            'provider': config['provider'],
            'base_url': config['base_url']
        }

        log_llm_request_response(prompt, config, error_info=error_info)

        provider_name = config['provider'].capitalize()
        print(f"Error calling {provider_name} API: {e}", file=sys.stderr)
        return None


def resume_session_with_prompt(session_id, prompt):
    """Resume Claude Code session and send a prompt."""
    if not session_id:
        print_log("Warning: No session ID found, skipping resume functionality")
        return False

    try:
        # Use Popen to start Claude Code with resume command
        process = subprocess.Popen(
            ['claude', '--dangerously-skip-permissions', '--resume', session_id],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True if sys.platform != "win32" else 0
        )

        # Send the prompt to Claude Code
        stdout, stderr = process.communicate(input=prompt, timeout=5)

        print_log(f"Successfully resumed session {session_id} and sent prompt")
        return True

    except subprocess.TimeoutExpired:
        # This is expected - Claude Code may keep running
        print_log(f"Session {session_id} resumed and prompt sent (continues in background)")
        return True


def main():
    """Main function to process hook content and extract Claude's latest response."""
    try:
        # Read hook content from stdin (Claude Code provides it this way)
        hook_content = sys.stdin.read()

        # Parse the hook content as JSON (json1)
        json1 = json.loads(hook_content.strip()) if hook_content.strip() else {}

        # Get hook event type from environment variable if available
        hook_event = os.environ.get('CLAUDE_HOOK_EVENT', 'unknown')


        # Ensure log directory exists
        script_dir = Path(__file__).parent.parent
        log_dir = script_dir / "log"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / "hook_content.log"

        # Prepare log entry with metadata
        timestamp = datetime.now().isoformat()
        separator = "\n" + "="*80 + "\n"

        # Extract transcript_path from json1
        transcript_path = json1.get('transcript_path')
        claude_response = None
        json2 = None
        session_id = None

        if transcript_path:
            transcript_path = Path(transcript_path)
            if transcript_path.exists():
                # Read JSONL file and get last line
                with open(transcript_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1].strip()
                        if last_line:
                            json2 = json.loads(last_line)
                            claude_response = extract_claude_response(json2)
                            session_id = extract_session_id(json2)
            else:
                raise FileNotFoundError(f"Transcript path does not exist: {transcript_path}")
        else:
            print_log("Warning: No transcript_path found in hook content")

        # Prepare log entry with formatted JSON outputs
        log_entry = f"""
{separator}
Timestamp: {timestamp}
Event Type: {hook_event}
Content Length: {len(hook_content)} characters

--- JSON1 (Hook Content) ---
{json.dumps(json1, indent=2, ensure_ascii=False)}
--- End JSON1 ---
"""

        if json2:
            log_entry += f"""
--- JSON2 (Last JSONL Entry) ---
{json.dumps(json2, indent=2, ensure_ascii=False)}
--- End JSON2 ---
"""

        if session_id:
            log_entry += f"""
--- Session ID ---
{session_id}
--- End Session ID ---
"""

        if claude_response:
            log_entry += f"""
--- Claude Code's Latest Response ---
{claude_response}
--- End Response ---
"""
        else:
            log_entry += """
--- Claude Code's Latest Response ---
No response found or could not extract Claude's response
--- End Response ---
"""

        print_log(log_entry)

        # Only log completion message to stderr
        print_log(f"Hook content logged to {log_file}")

        # Analyze Claude response and suggest next spec-kit command if session_id is found
        if session_id and claude_response:
            # Check if this session is within the trigger limit
            if check_and_increment_session_counter(session_id):
                # Load LLM configuration
                llm_config = load_llm_config()

                # Construct prompt using template
                analysis_prompt = construct_prompt_from_template(claude_response, llm_config['prompt_template'])
                if analysis_prompt:
                    print_log(f"Sending Claude response to LLM for analysis...")

                    # Log the full prompt being sent to LLM
                    prompt_log_entry = f"""
{separator}
Timestamp: {datetime.now().isoformat()}
--- LLM Analysis Prompt ---
{analysis_prompt}
--- End LLM Analysis Prompt ---
"""
                    with open(log_file, 'a', encoding='utf-8') as f:
                        f.write(prompt_log_entry)

                    # Get LLM suggestion
                    suggested_command = send_prompt_to_llm(analysis_prompt, llm_config)

                    # Log LLM response
                    if suggested_command:
                        print_log(f"LLM suggested command: {suggested_command}")

                        # Log the LLM response
                        llm_response_log_entry = f"""
{separator}
Timestamp: {datetime.now().isoformat()}
--- LLM Response ---
{suggested_command}
--- End LLM Response ---
"""
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(llm_response_log_entry)

                        # Sleep for 10 seconds before responding to Claude Code
                        print_log(f"Sleeping for 10 seconds before responding to Claude Code...")
                        time.sleep(10)

                        print_log(f"Attempting to resume session {session_id} with suggested command...")
                        resume_success = resume_session_with_prompt(session_id, suggested_command)
                        if not resume_success:
                            print_log(f"Failed to resume session {session_id}")
                    else:
                        # Log empty/failed LLM response
                        llm_response_log_entry = f"""
{separator}
Timestamp: {datetime.now().isoformat()}
--- LLM Response ---
Failed to get suggestion from LLM or empty response
--- End LLM Response ---
"""
                        with open(log_file, 'a', encoding='utf-8') as f:
                            f.write(llm_response_log_entry)

                        print_log("Failed to get suggestion from LLM")
                else:
                    print_log("Failed to construct analysis prompt")
            else:
                print_log(f"Session {session_id} has reached maximum trigger limit, not resuming")
        else:
            if not session_id:
                print_log("No session ID found, skipping resume functionality")
            if not claude_response:
                print_log("No Claude response found, skipping LLM analysis")

    except json.JSONDecodeError as e:
        print(f"Error parsing hook content as JSON: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error processing hook: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
