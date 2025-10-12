# service_demos/mcp_service.py
import os
import sys
import argparse
import json
import logging
import io
import socketserver
from threading import Lock

# --- Globals for the service ---
config = {}
core_lib = None
LOG_LEVEL = "INFO"
SERVICE_CONTRACT_TAG = None
print_lock = Lock() # To prevent interleaved prints from different threads

# --- The Core MCP Request Handler ---
class MCPRequestHandler(socketserver.BaseRequestHandler):
    """
    Handles incoming TCP requests for the MCP service.
    Each connection gets its own instance of this handler.
    """
    def handle(self):
        client_address = self.client_address[0]
        with print_lock:
            logging.info(f"Client connected: {client_address}")

        try:
            # Read a single line (the JSON request) from the client
            request_data = self.request.recv(4096).strip()
            if not request_data:
                return # Client connected and immediately disconnected

            request_json = json.loads(request_data.decode('utf-8'))
            command = request_json.get("command")

            if command == "LIST_AGENTS":
                response = self.handle_list_agents()
            elif command == "EXECUTE_AGENT":
                response = self.handle_execute_agent(request_json)
            else:
                response = self.build_error_response(f"Unknown command: '{command}'")

        except json.JSONDecodeError:
            response = self.build_error_response("Invalid JSON request.")
        except Exception as e:
            response = self.build_error_response(f"An unexpected server error occurred: {e}")
            logging.error(f"Error handling request from {client_address}: {e}", exc_info=True)
        
        # Send the JSON response back to the client, followed by a newline
        self.request.sendall((json.dumps(response) + '\n').encode('utf-8'))

        with print_lock:
            logging.info(f"Client disconnected: {client_address}")

    def handle_list_agents(self):
        """Builds a list of available agents based on the service contract."""
        available_agents = []
        for agent_name, agent_data in sorted(config.get('agents', {}).items()):
            if SERVICE_CONTRACT_TAG:
                if SERVICE_CONTRACT_TAG not in agent_data.get("web_services", []):
                    continue
            
            available_agents.append({
                "name": agent_name,
                "help": agent_data.get("help", ""),
                "inputs": agent_data.get("inputs", []),
                "optional_inputs": agent_data.get("optional_inputs", []),
                "outputs": agent_data.get("outputs", [])
            })
        return {"status": "SUCCESS", "payload": available_agents}

    def handle_execute_agent(self, request_json):
        """Executes an agent and captures its result, status, and log."""
        agent_name = request_json.get("agent")
        cli_args = request_json.get("params", {})

        agent_config = config.get('agents', {}).get(agent_name)
        if not agent_config:
            return self.build_error_response(f"Agent '{agent_name}' not found.")

        if SERVICE_CONTRACT_TAG:
            if SERVICE_CONTRACT_TAG not in agent_config.get("web_services", []):
                return self.build_error_response(f"Agent '{agent_name}' not found under this service contract.")

        # Capture logs for this specific execution
        log_stream = io.StringIO()
        request_log_handler = logging.StreamHandler(log_stream)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        request_log_handler.setFormatter(formatter)
        
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers = [request_log_handler]
        root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
        
        result, status = {}, {"status": {"value": 1, "reason": "Execution failed before start"}}
        try:
            # Call the core engine library function
            result, status = core_lib['exec_agent'](agent=agent_config, agent_name=agent_name, config=config, cli_args=cli_args, results={})
        except Exception as e:
            logging.error(f"Unhandled exception during agent '{agent_name}' execution: {e}", exc_info=True)
            status = {"status": {"value": 1, "reason": f"Unhandled server exception: {e}"}}
        finally:
            root_logger.handlers = original_handlers

        log_output = log_stream.getvalue()

        if status.get("status", {}).get("value", 1) == 0:
            return {
                "status": "SUCCESS",
                "payload": {
                    "agent": agent_name,
                    "result": result
                },
                "log": log_output
            }
        else:
            return {
                "status": "ERROR",
                "message": f"Agent '{agent_name}' failed to execute.",
                "details": status.get("status", {}),
                "log": log_output
            }

    def build_error_response(self, message):
        """Creates a standardized JSON error response."""
        return {"status": "ERROR", "message": message}

# --- Main Service Startup Logic ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A TCP-based MCP service for the Dynamic Agent Workflow engine.")
    parser.add_argument('--config', default='../config.json', help='Path to the configuration file.')
    parser.add_argument('--lib-path', default='..', required=True, help='Path to the directory containing dynamic_workflows_agents.py.')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the service to.')
    parser.add_argument('--port', type=int, default=5001, help='Port for the MCP service.')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level for the engine during agent execution.')
    parser.add_argument('--service', help='Activate a specific service contract by tag.')
    args = parser.parse_args()

    SERVICE_CONTRACT_TAG = args.service
    LOG_LEVEL = args.log_level
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [MCP SERVER] - %(levelname)s - %(message)s')

    if SERVICE_CONTRACT_TAG:
        logging.info(f"Service starting in 'Filtered Contract' mode. Activating contract tag: '{SERVICE_CONTRACT_TAG}'")
    else:
        logging.info("Service starting in 'Full Unfiltered' mode. All agents will be available.")
    
    try:
        with open(args.config, 'r') as f: config = json.load(f)
        logging.info(f"Configuration loaded from '{args.config}'.")
    except Exception as e:
        logging.error(f"FATAL: Could not load or parse config file '{args.config}': {e}")
        sys.exit(1)
        
    sys.path.insert(0, os.path.abspath(args.lib_path))
    try:
        import dynamic_workflows_agents
        core_lib = { "exec_agent": dynamic_workflows_agents.exec_agent, "setup_depth_manager": dynamic_workflows_agents.setup_depth_manager }
        logging.info("Successfully imported core workflow engine.")
    except ImportError:
        logging.error(f"FATAL: Could not import 'dynamic_workflows_agents.py'. Please check --lib-path.")
        sys.exit(1)

    core_lib['setup_depth_manager'](config)
    dynamic_workflows_agents.log_text_limit = int(config.get('workflow_settings', {}).get('log_text_limit', 500))
    logging.info("Core engine's depth manager initialized.")
    
    try:
        server = socketserver.ThreadingTCPServer((args.host, args.port), MCPRequestHandler)
        logging.info(f"MCP Agent Service starting on {args.host}:{args.port}")
        server.serve_forever()
    except Exception as e:
        logging.error(f"Could not start server: {e}")
    finally:
        if 'server' in locals():
            server.server_close()
            logging.info("Server shut down.")