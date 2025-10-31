
import os
import sys
import argparse
import json
import logging
import io # Import io for StringIO
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
import uvicorn

# --- Globals for the service ---
SERVICE_CONTRACT_TAG = None
LOG_LEVEL = "INFO"

# --- Create the FastAPI App ---
app = FastAPI(
    title="Dynamic Agent Workflow FastAPI Service",
    description="A standard, high-performance FastAPI service for executing agents from the Dynamic Agent Workflow engine.",
    version="1.0.0"
)

# --- API Endpoints ---

@app.get("/", summary="Health Check", tags=["General"])
def read_root():
    """
    Provides a simple health check to confirm the service is running.
    """
    return {"status": "online", "message": "FastAPI Agent Service is running."}

@app.get("/agents", summary="List Available Agents", tags=["General"])
def list_available_agents():
    """
    Returns a list of all agents available under the current service contract.
    """
    available_agents = []
    sorted_agent_names = sorted(config.get('agents', {}).keys())

    for agent_name in sorted_agent_names:
        agent_data = config['agents'][agent_name]
        
        # If a service contract is active, filter agents that don't have the tag.
        if SERVICE_CONTRACT_TAG:
            agent_contracts = agent_data.get("web_services", [])
            if SERVICE_CONTRACT_TAG not in agent_contracts:
                continue
        
        available_agents.append({
            "name": agent_name,
            "help": agent_data.get("help", "No help text provided."),
            "inputs": agent_data.get("inputs", []),
            "optional_inputs": agent_data.get("optional_inputs", []),
            "outputs": agent_data.get("outputs", [])
        })
    
    return {
        "service_contract": SERVICE_CONTRACT_TAG if SERVICE_CONTRACT_TAG else "full_unfiltered",
        "available_agents": available_agents
    }

@app.post("/execute/{agent_name:path}", summary="Execute an Agent", tags=["Agent Execution"])
async def execute_agent_endpoint(agent_name: str, request: Request):
    """
    Executes a specified agent by name.

    The request body must be a JSON object containing the parameters
    required by the agent's 'inputs' and 'optional_inputs'.
    """
    # The agent_name from the path will not have a leading slash, so add it back
    # to match the format in config.json.
    full_agent_name = f"/{agent_name}"

    agent_config = config.get('agents', {}).get(full_agent_name)
    if not agent_config:
        raise HTTPException(status_code=404, detail=f"Agent '{full_agent_name}' not found in the loaded configuration.")

    # Apply service contract filtering
    if SERVICE_CONTRACT_TAG:
        agent_contracts = agent_config.get("web_services", [])
        if SERVICE_CONTRACT_TAG not in agent_contracts:
            raise HTTPException(status_code=404, detail=f"Agent '{full_agent_name}' not found under the active service contract '{SERVICE_CONTRACT_TAG}'.")

    if not core_lib:
        raise HTTPException(status_code=503, detail="Core engine library is not loaded. Check server logs for errors.")

    try:
        cli_args = await request.json()
        if not isinstance(cli_args, dict):
            raise HTTPException(status_code=400, detail="Request body must be a JSON object.")
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid or empty JSON in request body.")

    # --- Log Capture Setup ---
    log_stream = io.StringIO()
    request_log_handler = logging.StreamHandler(log_stream)
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    request_log_handler.setFormatter(formatter)
    
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:] # Save original handlers
    root_logger.handlers = [request_log_handler] # Redirect to StringIO
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    # --- End Log Capture Setup ---

    # Each request gets a fresh results tape.
    results_tape = {}
    
    result, status = {}, {"status": {"value": 1, "reason": "Execution failed before start"}}
    try:
        # Execute the agent using the dynamically loaded core library function.
        result, status = core_lib['exec_agent'](
            agent=agent_config,
            agent_name=full_agent_name,
            config=config,
            cli_args=cli_args,
            results=results_tape
        )
        
    except Exception as e:
        logging.error(f"An unhandled exception occurred during agent '{full_agent_name}' execution: {e}", exc_info=True)
        status = {"status": {"value": 1, "reason": f"Unhandled server exception: {e}"}}
    finally:
        # --- Log Capture Teardown ---
        root_logger.handlers = original_handlers # Restore original handlers
        log_output = log_stream.getvalue()
        # --- End Log Capture Teardown ---

    # Return a structured JSON response with the outcome and logs.
    return {
        "agent_name": full_agent_name,
        "execution_status": status,
        "final_result": result,
        "execution_log": log_output # Include the captured logs
    }


# --- Main Service Startup Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A FastAPI web service for the Dynamic Agent Workflow engine.")
    parser.add_argument('--config', default='../config.json', help='Path to the configuration file (default: ../config.json).')
    parser.add_argument('--lib-path', required=True, help='Path to the directory containing dynamic_workflows_agents.py.')
    parser.add_argument('--host', default='127.0.0.1', help='Host to bind the web service to (default: 127.0.0.1).')
    parser.add_argument('--port', type=int, default=8000, help='Port to run the web service on (default: 8000).')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level for the engine (default: INFO).')
    parser.add_argument('--service', help='Activate a specific service contract by tag. Only agents with this tag in their `web_services` list will be exposed.')
    args = parser.parse_args()

    SERVICE_CONTRACT_TAG = args.service
    LOG_LEVEL = args.log_level

    logging.basicConfig(level=getattr(logging, LOG_LEVEL), format='%(asctime)s - [FastAPI SERVER] - %(levelname)s - %(message)s')

    if SERVICE_CONTRACT_TAG:
        logging.info(f"Service starting in 'Filtered Contract' mode. Activating contract tag: '{SERVICE_CONTRACT_TAG}'")
    else:
        logging.info("Service starting in 'Full Unfiltered' mode. All agents will be available.")

    # --- Load Configuration ---
    try:
        with open(args.config, 'r') as f:
            config = json.load(f)
        logging.info(f"Configuration loaded from '{args.config}'.")
    except Exception as e:
        logging.error(f"FATAL: Could not load or parse config file '{args.config}': {e}")
        sys.exit(1)
        
    # --- Dynamically Load Core Library ---
    # This allows the service to run from anywhere without installing the core engine as a package.
    sys.path.insert(0, os.path.abspath(args.lib_path))
    try:
        import dynamic_workflows_agents
        core_lib = { "exec_agent": dynamic_workflows_agents.exec_agent, "setup_depth_manager": dynamic_workflows_agents.setup_depth_manager }
        logging.info("Successfully imported core workflow engine.")
    except ImportError:
        logging.error(f"FATAL: Could not import 'dynamic_workflows_agents.py'. Please check the --lib-path argument.")
        sys.exit(1)

    # --- Initialize Core Engine ---
    core_lib['setup_depth_manager'](config)
    dynamic_workflows_agents.log_text_limit = int(config.get('workflow_settings', {}).get('log_text_limit', 500))
    logging.info("Core engine's depth manager initialized.")
    
    logging.info(f"Starting FastAPI agent service on http://{args.host}:{args.port}")
    logging.info(f"View interactive API docs at http://{args.host}:{args.port}/docs")

    # --- Run the Uvicorn Server ---
    uvicorn.run(app, host=args.host, port=args.port)

