# --- START OF FILE server_demo.py ---

import json
import logging
import logging.handlers
import sys
import argparse
import os

# --- COMMAND-LINE ARGUMENT PARSING & DYNAMIC LIBRARY LOADING ---
# This section is added to make the demo flexible and robust.

# 1. DEFINE AND PARSE ARGUMENTS
parser = argparse.ArgumentParser(
    description="Headless Workflow Execution Engine Demonstration.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    '--config',
    default='config.json',
    help='Path to the configuration file to use (default: config.json)'
)
parser.add_argument(
    '--lib-path',
    help='Path to the directory containing the dynamic_workflows_agents.py core library.'
)
args = parser.parse_args()

# 2. VALIDATE CONFIG FILE PATH
if not os.path.isfile(args.config):
    print(f"FATAL ERROR: Configuration file not found at '{os.path.abspath(args.config)}'.\n")
    parser.print_help()
    exit(1)

# 3. PREPARE LIBRARY PATH
if args.lib_path:
    sys.path.insert(0, os.path.abspath(args.lib_path))

# 4. IMPORT CORE LIBRARY WITH CORRECT ERROR HANDLING
try:
    import dynamic_workflows_agents
    from dynamic_workflows_agents import exec_agent, setup_depth_manager
except ImportError:
    print("FATAL ERROR: Could not import the core workflow engine from 'dynamic_workflows_agents.py'.\n")
    parser.print_help()
    exit(1)

# --- END OF NEW SECTION ---


"""
server.py - Headless Workflow Execution Engine Demonstration

This script serves as a proof-of-concept and template for using the
dynamic_workflows_agents.py engine as an embeddable library within a larger
application, such as a network service.

It demonstrates the core pattern required for "headless" execution:
  1. Importing the necessary functions directly from the core engine.
  2. Bypassing all command-line interface (CLI) parsing logic.
  3. Programmatically loading the configuration.
  4. Setting up a dedicated logging system for the host application.
  5. Manually preparing the agent and input data, simulating what a network
     listener would do with an incoming request payload.
  6. Calling the exec_agent() function directly with in-memory objects.
  7. Processing the results, which in a real service would be serialized and
     returned to a client.

This pattern proves the core engine is fully decoupled and can be integrated
into a wide variety of hosting applications.

-----------------------------------------------------------------------------
        Potential Real-World Network Service Implementations
-----------------------------------------------------------------------------

This foundational script can be extended to create robust, scalable services:

1.  **RESTful API Server (using Flask/FastAPI):**
    -   Create an endpoint like `POST /execute/{agent_name}`.
    -   The request body (JSON) would contain the workflow inputs.
    -   The server would run the specified agent and return the workflow's
      final result as a JSON response.
    -   Use cases: Triggering complex business logic from webhooks, providing
      a microservice for data transformation, running reports on demand.

2.  **Message Queue Consumer (using RabbitMQ/Kafka):**
    -   The server would listen to a specific queue or topic for incoming messages.
    -   Each message would contain the agent name and its inputs.
    -   The server would execute the workflow for each message, potentially
      asynchronously.
    -   The result could be published to another queue for further processing.
    -   Use cases: Real-time event processing, ETL pipelines, parallel
      task execution for high-throughput data streams.

3.  **Scheduled Task Runner (using APScheduler/Celery Beat):**
    -   The server would run on a schedule (e.g., every 5 minutes, once a day).
    -   On schedule, it would execute a predefined maintenance or reporting workflow.
    -   Use cases: Generating daily reports, data cleanup tasks, system
      health checks, database synchronization.

4.  **Interactive REPL / Admin Shell:**
    -   Build an interactive command-line shell for administrators to manually
      trigger and debug workflows on a live server.
    -   Provides powerful, direct access for diagnostics and manual overrides.

5.  **TCP/UDP Socket Server:**
    -   For specialized, high-performance, or legacy integrations, the server
      could listen on a raw socket for specific data protocols (e.g., HL7, FIX).
    -   Each incoming data packet would trigger a workflow to parse, process,
      and route the information.
"""

def setup_server_logging():
    """Sets up a dedicated logger for the server, independent of the CLI."""
    logger = logging.getLogger()
    
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logger.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        '%(asctime)s - [SERVER] - %(levelname)s - %(message)s'
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logging.info("Server logging configured.")

def load_config(config_path: str):
    """Loads the agent configuration file directly."""
    try:
        with open(config_path, 'r') as f:
            logging.info(f"Configuration loaded from {config_path}")
            return json.load(f)
    except FileNotFoundError:
        logging.error(f"FATAL: Configuration file '{config_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        logging.error(f"FATAL: Configuration file '{config_path}' is not a valid JSON.")
        sys.exit(1)

def get_agent_for_execution(agent_name: str, config: dict):
    """
    Selects an agent from the configuration.
    """
    agent_config = config['agents'].get(agent_name)
    if not agent_config:
        logging.error(f"Agent '{agent_name}' not found in configuration.")
        return None

    return agent_config

def main_server_test():
    """Main function to demonstrate a headless workflow execution."""
    
    setup_server_logging()
    # Use the config path from the command-line arguments
    config = load_config(config_path=args.config)

    # --- Define the workflow to run ---
    agent_name_to_run = "loop_start_beta"
    workflow_inputs = {
        "step_index": 0,
        "loop_max": 5
    }

    # --- Prepare the execution environment ---
    
    setup_depth_manager(config)
    
    dynamic_workflows_agents.log_text_limit = int(config.get('workflow_settings', {}).get('log_text_limit', 1024))
    
    agent_to_execute = get_agent_for_execution(agent_name_to_run, config)

    if not agent_to_execute:
        logging.error("Could not prepare agent for execution. Aborting.")
        return

    # --- Execute the workflow ---
    logging.info(f"--- Starting Headless Workflow: {agent_name_to_run} ---")
    results_tape = {} 
    
    try:
        final_result, status = exec_agent(
            agent=agent_to_execute,
            agent_name=agent_name_to_run,
            config=config,
            cli_args=workflow_inputs,
            results=results_tape
        )

        # --- Process the results ---
        logging.info("--- Workflow Execution Complete ---")
        print("\n--- Final Result ---")
        print(json.dumps(final_result, indent=2))
        
        print("\n--- Final Status ---")
        print(json.dumps(status, indent=2))
        
        if status['status']['value'] != 0:
            logging.error(f"Workflow finished with an error: {status['status']['reason']}")
        else:
            logging.info("Workflow finished successfully.")

    except Exception as e:
        logging.error(f"An unhandled exception occurred during workflow execution: {e}", exc_info=True)

if __name__ == "__main__":
    main_server_test()