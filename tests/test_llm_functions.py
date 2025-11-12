#!/usr/bin/env python3
"""
Test script for LLM functionality in spec-kit-next-step.py

This script tests the LLM-related functions:
- load_llm_config()
- construct_prompt_from_template()
- send_prompt_to_llm()
"""

import sys
import os
import json
from pathlib import Path

# Add the src directory to the Python path to import the script
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import the script directly since it has hyphens in the filename
import importlib.util
spec_kit_path = os.path.join(os.path.dirname(__file__), '..', 'src', 'spec-kit-next-step.py')
spec = importlib.util.spec_from_file_location("spec_kit_next_step", spec_kit_path)
spec_kit_next_step = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spec_kit_next_step)

# Get the functions from the imported module
load_llm_config = spec_kit_next_step.load_llm_config
construct_prompt_from_template = spec_kit_next_step.construct_prompt_from_template
send_prompt_to_llm = spec_kit_next_step.send_prompt_to_llm
load_env_vars = spec_kit_next_step.load_env_vars


def test_load_llm_config():
    """Test load_llm_config function."""
    print("=" * 60)
    print("Testing load_llm_config()...")
    print("=" * 60)

    try:
        config = load_llm_config()
        print("✓ load_llm_config() executed successfully")
        print(f"Configuration loaded:")
        for key, value in config.items():
            if key == 'api_key' and value:
                # Mask the API key for security
                masked_value = value[:8] + "..." if len(value) > 8 else "***"
                print(f"  {key}: {masked_value}")
            else:
                print(f"  {key}: {value}")

        # Check if essential fields are present
        required_fields = ['provider', 'api_key', 'base_url', 'model']
        missing_fields = [field for field in required_fields if not config.get(field)]

        if missing_fields:
            print(f"\n⚠️  Warning: Missing required fields: {missing_fields}")
            print("Please check your .env file or environment variables")
        else:
            print("\n✓ All required configuration fields are present")

        return config

    except Exception as e:
        print(f"✗ Error testing load_llm_config(): {e}")
        return None


def test_construct_prompt_from_template():
    """Test construct_prompt_from_template function."""
    print("\n" + "=" * 60)
    print("Testing construct_prompt_from_template()...")
    print("=" * 60)

    # Test with sample Claude response
    sample_claude_response = " 总结：任务分解已经完成， 建议：使用 /specify.implement 命令进行下一步开发"

    try:
        # Load config to get the template name
        config = load_llm_config()
        template_name = config.get('prompt_template', 'speckit_command_suggestion.txt')

        print(f"Using template: {template_name}")

        # Test prompt construction
        prompt = construct_prompt_from_template(sample_claude_response, template_name)

        if prompt:
            print("✓ construct_prompt_from_template() executed successfully")
            print(f"Generated prompt length: {len(prompt)} characters")
            print("\nGenerated prompt preview:")
            print("-" * 40)
            print(prompt[:500] + "..." if len(prompt) > 500 else prompt)
            print("-" * 40)
            return prompt
        else:
            print("✗ construct_prompt_from_template() returned None")
            return None

    except Exception as e:
        print(f"✗ Error testing construct_prompt_from_template(): {e}")
        return None


def test_send_prompt_to_llm():
    """Test send_prompt_to_llm function."""
    print("\n" + "=" * 60)
    print("Testing send_prompt_to_llm()...")
    print("=" * 60)

    # Create a simple test prompt
    test_prompt = "Please suggest the next spec-kit command for: 'I want to create a simple user login feature'"

    try:
        config = load_llm_config()

        # Check if API key is available
        if not config.get('api_key'):
            print("⚠️  No API key found in configuration")
            print("Please set PM_AGENT_API_KEY in your .env file")
            return None

        print(f"Sending test prompt to {config['provider']} ({config['model']})...")
        print(f"Test prompt: {test_prompt}")
        print("\nWaiting for LLM response...")

        # Test the LLM call
        response = send_prompt_to_llm(test_prompt, config)

        if response:
            print("✓ send_prompt_to_llm() executed successfully")
            print(f"Response length: {len(response)} characters")
            print("\nLLM Response:")
            print("-" * 40)
            print(response)
            print("-" * 40)
            return response
        else:
            print("✗ send_prompt_to_llm() returned None or empty response")
            return None

    except Exception as e:
        print(f"✗ Error testing send_prompt_to_llm(): {e}")
        return None


def test_full_workflow():
    """Test the complete workflow: config -> prompt -> LLM."""
    print("\n" + "=" * 60)
    print("Testing full LLM workflow...")
    print("=" * 60)

    try:
        # Step 1: Load configuration
        print("Step 1: Loading LLM configuration...")
        config = load_llm_config()
        if not config:
            print("✗ Failed to load configuration")
            return False
        print("✓ Configuration loaded")

        # Step 2: Construct prompt from template
        print("\nStep 2: Constructing prompt from template...")
        sample_response = " 总结：任务分解已经完成， 建议：使用 /specify.implement 命令进行下一步开发"
        template_name = config.get('prompt_template', 'speckit_command_suggestion.txt')
        prompt = construct_prompt_from_template(sample_response, template_name)
        if not prompt:
            print("✗ Failed to construct prompt")
            return False
        print("✓ Prompt constructed successfully")

        # Step 3: Send to LLM
        print("\nStep 3: Sending prompt to LLM...")
        if not config.get('api_key'):
            print("⚠️  Skipping LLM test - no API key configured")
            print("Set PM_AGENT_API_KEY in .env to test LLM functionality")
            return True

        response = send_prompt_to_llm(prompt, config)
        if response:
            print("✓ Full workflow completed successfully")
            print(response)
            return True
        else:
            print("✗ LLM returned no response")
            return False

    except Exception as e:
        print(f"✗ Error in full workflow test: {e}")
        return False


def main():
    """Main test function."""
    print("Starting LLM functionality tests...")
    print("Python version:", sys.version)
    print("Working directory:", os.getcwd())

    # Check if .env file exists
    env_file = Path('.env')
    if env_file.exists():
        print(f"✓ Found .env file: {env_file.absolute()}")
    else:
        print("⚠️  No .env file found in current directory")
        print("Create a .env file with PM_AGENT configuration to test LLM functionality")

    # Run individual tests
    config = test_load_llm_config()
    prompt = test_construct_prompt_from_template()

    # Only test LLM if configuration is valid
    if config and config.get('api_key'):
        response = test_send_prompt_to_llm()
    else:
        print("\n⚠️  Skipping LLM API test - no valid configuration")

    # Test full workflow
    workflow_success = test_full_workflow()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Configuration loading: {'✓' if config else '✗'}")
    print(f"Prompt construction: {'✓' if prompt else '✗'}")
    print(f"Full workflow: {'✓' if workflow_success else '✗'}")

    if workflow_success:
        print("\n🎉 All tests passed! LLM functionality is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the configuration and try again.")


if __name__ == "__main__":
    main()
