import json
import logging
import logging.handlers
import sys
import dynamic_workflows_agents  # Import the module itself

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

# Import the necessary functions from the workflow engine
from dynamic_workflows_agents import (
    exec_agent, 
    setup_depth_manager
)

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

def load_config(config_path: str = 'config.json'):
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

def get_agent_for_execution(agent_name: str, agent_inputs: dict, config: dict):
    """
    Selects an agent and promotes it to a temporary workflow if it's not already one.
    """
    agent_config = config['agents'].get(agent_name)
    if not agent_config:
        logging.error(f"Agent '{agent_name}' not found in configuration.")
        return None

    return agent_config

import time

import time
import logging # Need to import logging to control it

def main_server_test():
    """
    Main function to run a performance test of the headless workflow engine.
    """
    
    # --- SETUP PHASE ---
    setup_start_time = time.perf_counter()
    
    setup_server_logging()
    config = load_config()

    agent_name_to_run = "loop_start_beta"
    workflow_inputs = {
        "step_index": 0,
        "loop_max": 5
    }
    
    setup_depth_manager(config)
    
    import dynamic_workflows_agents
    dynamic_workflows_agents.log_text_limit = int(config.get('workflow_settings', {}).get('log_text_limit', 1024))
    
    agent_to_execute = get_agent_for_execution(agent_name_to_run, workflow_inputs, config)

    if not agent_to_execute:
        logging.error("Could not prepare agent for execution. Aborting.")
        return

    setup_end_time = time.perf_counter()
    setup_duration = setup_end_time - setup_start_time
    logging.info(f"Setup and configuration loading complete in {setup_duration:.4f} seconds.")

    # --- EXECUTION PHASE ---
    logging.info("Starting performance test (disabling verbose logging for accuracy)...")
    
    # Temporarily disable all logging below CRITICAL to prevent I/O overhead
    logging.getLogger().setLevel(logging.CRITICAL)
    
    num_runs = 1000
    execution_start_time = time.perf_counter()

    try:
        for i in range(num_runs):
            results_tape = {} 
            
            final_result, status = exec_agent(
                agent=agent_to_execute,
                agent_name=agent_name_to_run,
                config=config,
                cli_args=workflow_inputs,
                results=results_tape
            )

    except Exception as e:
        # Re-enable logging before printing the error
        logging.getLogger().setLevel(logging.DEBUG)
        logging.error(f"An unhandled exception occurred during execution loop: {e}", exc_info=True)
        return
    
    execution_end_time = time.perf_counter()
    
    # Re-enable logging for the final report
    logging.getLogger().setLevel(logging.DEBUG)
    
    execution_duration = execution_end_time - execution_start_time
    total_duration = execution_end_time - setup_start_time
    
    # --- REPORTING PHASE ---
    
    avg_time_per_event = (execution_duration / num_runs) * 1000  # in milliseconds
    
    # Use logging for the final report for consistency
    logging.info("\n" + "="*50 +
                 "\n           PERFORMANCE TEST SUMMARY" +
                 "\n" + "="*50 +
                 f"\nInitial Setup Time:       {setup_duration:.4f} seconds" +
                 f"\nTotal Execution Time:     {execution_duration:.4f} seconds (for {num_runs} events)" +
                 f"\nAverage Time per Event:   {avg_time_per_event:.4f} ms" +
                 f"\nThroughput:               {num_runs / execution_duration:.2f} events/sec" +
                 f"\nTotal Test Duration:      {total_duration:.4f} seconds" +
                 "\n" + "="*50)

if __name__ == "__main__":
    main_server_test()
