# Agent Configuration Diff Tool (`config_diff.py`)

`config_diff.py` is a powerful command-line utility designed to intelligently compare two `config.json` files from the Dynamic Agent Workflow framework.  It is in **utilities/config_diff.py** because it is a support tool, not part of the core. It is strategically stepping out of the GUI to rapidly manage the json files with a context aware tool that is agent and tag aware inside those agents. 

Unlike a standard text `diff` tool, this script understands the structure of the agent configuration. It provides a clean, actionable report that highlights differences in a way that is immediately useful for developers managing multiple versions of their agent library. It is an essential tool for synchronizing configurations between development, testing, and production environments.

As one user reported after its first use:
> **"I just unified 3 config files into one config file in minutes not hours."**

## Features

*   **Identifies Unique Agents:** Reports which agents exist in one file but are missing from the other.
*   **Copy-Paste Ready Output:** Formats unique agent definitions as complete, valid JSON blocks that can be copied and pasted directly into another config file.
*   **Detailed Content Comparison:** For agents that exist in both files, it pinpoints exactly which top-level keys (like `help`, `function_def`, or `steps`) have different content.
*   **Copy-Pasteable Fragments:** Displays the differing content as JSON fragments, making it easy to copy a specific correction from one file to another.
*   **Simple Command-Line Interface:** Easy to integrate into your development and release management workflow.

## Usage

The script is run from the command line, taking the paths to the two configuration files as arguments.

```bash
python config_diff.py <path_to_first_config.json> <path_to_second_config.json>
```

### Example

Imagine you have your main `config.json` and a `config.release.json` from a different branch.

```bash
python config_diff.py config.json config.release.json
```

## Sample Output

The tool generates a three-part report directly to your console.

```
--- Agents found only in first config file ---

(None)


--- Agents found only in second config file ---

"sftp_get": {
  "type": "proc",
  "help": "The correct way to download a file. Securely downloads a file's raw bytes...",
  "inputs": [
    "hostname",
    "username",
    "password",
    "remote_path"
  ],
  "optional_inputs": [],
  "outputs": [
    "file_content"
  ],
  "function": "sftp_get",
  "function_def": "import paramiko..."
},


--- Agents with different content (found in both files) ---

--- Agent 'append_text' differs ---

First config file has this value for 'web_services':
"web_services": [
  "public_api",
  "test_add"
],

Second config file has this value for 'web_services':
"web_services": [
  "public_api",
  "internal_tools"
],

--- Agent 'extract_json' differs ---

First config file has this value for 'function_def':
"function_def": "import json_repair...",

Second config file has this value for 'function_def':
"function_def": "import json\nimport json_repair\nimport re...",
```

### How to Interpret the Output

*   **"Agents found only in..." Sections:** These sections show you complete agent definitions that are missing from the other file. You can highlight, copy, and paste the entire block to synchronize your files. The tool automatically handles the correct JSON formatting, including the trailing comma.
*   **"Agents with different content..." Section:** This is the most powerful part for managing configuration drift. It pinpoints the exact agents and the specific keys within them that have changed, allowing you to quickly resolve conflicts and ensure consistency across your environments.