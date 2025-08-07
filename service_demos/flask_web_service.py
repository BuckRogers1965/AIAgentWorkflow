# agent_service.py
import os
import sys
import argparse
import json
import logging
import io
import time
from flask import Flask, request, Response
import xml.etree.ElementTree as ET
from xml.dom import minidom

# --- Globals for the service ---
config = {}
core_lib = None
LOG_LEVEL = "INFO"
SERVICE_CONTRACT_TAG = None

# --- Create the Flask App ---
app = Flask(__name__)

# --- XML Helper Functions (No changes) ---
def dict_to_xml(parent_element, dictionary):
    for key, value in dictionary.items():
        element_name = str(key).replace(" ", "_")
        if isinstance(value, dict):
            sub_element = ET.SubElement(parent_element, element_name)
            dict_to_xml(sub_element, value)
        elif isinstance(value, list):
            list_element = ET.SubElement(parent_element, element_name)
            for item in value:
                item_element = ET.SubElement(list_element, "item")
                if isinstance(item, dict):
                     dict_to_xml(item_element, item)
                else:
                    item_element.text = str(item)
        else:
            sub_element = ET.SubElement(parent_element, element_name)
            sub_element.text = str(value)

def build_error_xml(message, status_code):
    root = ET.Element("error")
    ET.SubElement(root, "code").text = str(status_code)
    ET.SubElement(root, "message").text = message
    xml_str = ET.tostring(root, 'utf-8')
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    return Response(pretty_xml_str, mimetype='application/xml', status=status_code)

# --- Agent Discovery Endpoint (No changes) ---
@app.route('/', methods=['GET'])
def list_available_agents():
    root = ET.Element("availableAgents")
    root.set("serviceName", "Dynamic Agent Workflow Service")
    if SERVICE_CONTRACT_TAG:
        root.set("activeContract", SERVICE_CONTRACT_TAG)
    else:
        root.set("activeContract", "full_unfiltered")

    sorted_agent_names = sorted(config.get('agents', {}).keys())

    for agent_name in sorted_agent_names:
        agent_data = config['agents'][agent_name]
        if SERVICE_CONTRACT_TAG:
            agent_contracts = agent_data.get("web_services", [])
            if SERVICE_CONTRACT_TAG not in agent_contracts:
                continue
        
        agent_element = ET.SubElement(root, "agent")
        ET.SubElement(agent_element, "name").text = agent_name
        ET.SubElement(agent_element, "help").text = agent_data.get("help", "No help text provided.")
        inputs_element = ET.SubElement(agent_element, "inputs")
        for input_param in agent_data.get("inputs", []):
            ET.SubElement(inputs_element, "param").text = input_param
        optional_inputs_element = ET.SubElement(agent_element, "optionalInputs")
        for opt_param in agent_data.get("optional_inputs", []):
            ET.SubElement(optional_inputs_element, "param").text = opt_param
        outputs_element = ET.SubElement(agent_element, "outputs")
        for output_param in agent_data.get("outputs", []):
            ET.SubElement(outputs_element, "param").text = output_param

    xml_str = ET.tostring(root, 'utf-8')
    pretty_xml_str = minidom.parseString(xml_str).toprettyxml(indent="  ")
    return Response(pretty_xml_str, mimetype='application/xml')


# --- Core Execution Endpoint (MODIFIED) ---
@app.route('/execute/<string:agent_name>', methods=['GET'])
def execute_agent_endpoint(agent_name):
    agent_config = config.get('agents', {}).get(agent_name)
    if not agent_config:
        return build_error_xml(f"Agent '{agent_name}' not found.", 404)

    if SERVICE_CONTRACT_TAG:
        agent_contracts = agent_config.get("web_services", [])
        if SERVICE_CONTRACT_TAG not in agent_contracts:
            return build_error_xml(f"Agent '{agent_name}' not found.", 404)

    if not core_lib:
        return build_error_xml("Core engine not loaded. Check server logs.", 503)

    cli_args = request.args.to_dict()
    log_stream = io.StringIO()
    request_log_handler = logging.StreamHandler(log_stream)
    request_log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
    root_logger = logging.getLogger()
    original_handlers = root_logger.handlers[:]
    root_logger.handlers = [request_log_handler]
    root_logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))
    
    result, status = {}, {"status": {"value": -1, "reason": "Execution failed before start"}}
    try:
        result, status = core_lib['exec_agent'](agent=agent_config, agent_name=agent_name, config=config, cli_args=cli_args, results={})
    except Exception as e:
        logging.error(f"Unhandled exception during agent execution: {e}", exc_info=True)
        status = {"status": {"value": 1, "reason": f"Unhandled server exception: {e}"}}
    finally:
        root_logger.handlers = original_handlers

    log_output = log_stream.getvalue()
    root = ET.Element("agentResponse")
    ET.SubElement(root, "agent").text = agent_name
    status_element = ET.SubElement(root, "status")
    dict_to_xml(status_element, status.get("status", {}))
    results_element = ET.SubElement(root, "results")
    if isinstance(result, dict):
        dict_to_xml(results_element, result)
    else:
        raw_result_element = ET.SubElement(results_element, "rawOutput")
        if isinstance(result, bytes):
            try: raw_result_element.text = result.decode('utf-8')
            except UnicodeDecodeError: raw_result_element.text = f"Non-UTF8-decodable bytes, length: {len(result)}"
        else: raw_result_element.text = str(result)
    
    # --- THIS IS THE FIX ---
    # The standard ElementTree library doesn't have a CDATA object.
    # We must manually construct it in the final string output.
    # We will use a unique placeholder that we can replace later.
    log_element = ET.SubElement(root, "log")
    cdata_placeholder = "[[CDATA_PLACEHOLDER]]"
    log_element.text = cdata_placeholder
    
    # First, convert the ElementTree object to a string
    xml_str = ET.tostring(root, 'utf-8', method='xml').decode('utf-8')
    
    # Now, replace the placeholder with the actual CDATA block
    final_xml_str = xml_str.replace(cdata_placeholder, f"<![CDATA[{log_output}]]>")
    
    # Prettify and return the final XML
    pretty_xml_str = minidom.parseString(final_xml_str).toprettyxml(indent="  ")
    return Response(pretty_xml_str, mimetype='application/xml')
    # --- END OF FIX ---


# --- Main Service Startup Logic (No changes) ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A Flask web service for the Dynamic Agent Workflow engine.")
    parser.add_argument('--config', default='config.json', help='Path to the configuration file.')
    parser.add_argument('--lib-path', required=True, help='Path to the directory containing dynamic_workflows_agents.py.')
    parser.add_argument('--host', default='0.0.0.0', help='Host to bind the web service to.')
    parser.add_argument('--port', type=int, default=5000, help='Port to run the web service on.')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], default='INFO', help='Logging level for the engine.')
    parser.add_argument('--service', help='Activate a specific service contract by tag.')
    args = parser.parse_args()

    SERVICE_CONTRACT_TAG = args.service
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - [SERVER] - %(levelname)s - %(message)s')
    LOG_LEVEL = args.log_level

    if SERVICE_CONTRACT_TAG:
        logging.info(f"Service starting in 'Filtered Contract' mode. Activating contract tag: '{SERVICE_CONTRACT_TAG}'")
    else:
        logging.info("Service starting in 'Full Unfiltered' mode. All agents will be available.")
    
    try:
        with open(args.config, 'r') as f: config = json.load(f)
        logging.info(f"Configuration loaded from '{args.config}'.")
    except Exception as e:
        logging.error(f"FATAL: Could not load or parse config file '{args.goconfi}': {e}")
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
    logging.info(f"Starting agent service on http://{args.host}:{args.port}")


    # to suppress the noisy INFO-level access logs.
    logging.getLogger('werkzeug').setLevel(logging.ERROR)
    # --- END OF FIX ---

    app.run(host=args.host, port=args.port)
