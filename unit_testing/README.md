# Unit Testing Framework


added command line options.  

To run this from this directory in the git hub project.

'''bash
    unittest % python test_runner.py --config ../config.json --lib-path ..
'''


'''
unittest % python test_runner.py --help                               
usage: test_runner.py [-h] [--config CONFIG] [--lib-path LIB_PATH] [--loglevel {DEBUG,INFO,WARNING,ERROR}]

Agent Workflow Test Runner. Scans config.json for agents with saved unit tests and executes them, generating an HTML report.

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the configuration file (default: config.json)
  --lib-path LIB_PATH   Path to the directory containing the dynamic_workflows_agents.py core library.
  --loglevel {DEBUG,INFO,WARNING,ERROR}
                        Set the logging level for capturing logs on FAILED tests.
                        - DEBUG: Most verbose, shows all steps.
                        - INFO: Shows standard execution flow (default).
                        - WARNING: Shows only warnings and errors.
                        - ERROR: Shows only fatal errors.
'''

This directory contains the Quality Assurance (QA) service for the Dynamic Agent Workflow platform. The `test_runner.py` script is a powerful command-line tool that automates the testing of every agent defined in your `config.json`, ensuring the reliability and correctness of your entire workflow library.

## Core Philosophy: Tests Live with the Code

The testing framework follows the platform's core principle of **absolute encapsulation**. A test case is not a separate entity in a different file; it is an integral part of the agent's definition itself.

This is achieved via the `run_config` block within an agent's JSON definition. This block allows an agent to be self-describing about how it should be tested.

**Example `run_config` in `config.json`:**
```json
"my_agent/v1.0": {
  "type": "proc",
  // ... other definitions ...
  "run_config": {
    "last_inputs": {
      "user_id": "test_user_123",
      "mode": "dryrun"
    },
    "tests": [
      {
        "output_variable": "validation_message",
        "assertion_type": "Equals",
        "expected_value": "User is valid."
      },
      {
        "output_variable": "status.value",
        "assertion_type": "Equals",
        "expected_value": "0"
      }
    ]
  }
}
```

## The Test Runner (`test_runner.py`)

This script is the engine of the QA service. It performs the following steps:
1.  Parses the specified `config.json` file.
2.  Scans for all agents that contain a `run_config` block with a `tests` array.
3.  For each testable agent, it executes the agent using the `last_inputs` provided.
4.  It then runs all defined assertions, comparing the actual results from the execution against the `expected_value`.
5.  Finally, it compiles all results into a single, professional, self-contained HTML report.

### How to Use

The test runner is designed to be run from the command line.

1.  **Navigate to the `unit_testing` directory.**
2.  **Run the script, pointing it to your main `config.json` file.**

```bash
# From the root of the project
python unit_testing/test_runner.py --config config.json
```
_Note: If you run it from within the `unit_testing` directory, you'll need to use `../config.json`._

#### Command-Line Arguments

*   `--config <path>`: (Required) The path to the `config.json` file you want to test. Default: `config.json`.
*   `--loglevel <LEVEL>`: (Optional) Set the log verbosity (`DEBUG`, `INFO`, `WARNING`, `ERROR`) to be captured for **failed tests**. `DEBUG` is highly recommended for diagnosing failures. Default: `INFO`.

### The HTML Report

The runner will generate a timestamped HTML file (e.g., `Test-Report-2023-10-28_12-00-00.html`). This report provides:
*   A high-level summary of the test run (total tests, pass/fail counts).
*   A collapsible, hierarchical view of all tested agents.
*   Clear `PASS` / `FAIL` indicators for every single assertion.
*   **Rich diagnostics for failures**, including the captured execution log and a full JSON dump of the final results tape.

This service is essential for maintaining a stable and reliable library of agents, especially in a collaborative or production environment. It provides the confidence to refactor and expand the system without introducing regressions.
