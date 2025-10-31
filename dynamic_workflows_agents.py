#!/home/jrogers/Documents/ai/chat/ragserver/venv/bin/python3

import re
import time
import base64
import logging
import logging.handlers
import json
from typing import Union, Dict, Any

_proc_agent_namespace = globals().copy()  # create a sandbox name space for proc agent functions

import os
import copy
import sys
import argparse

import traceback

# --- Conditionally load the sandbox module ---
try:
    from sandbox import sandbox_module
    SANDBOX_MODULE_AVAILABLE = sandbox_module.RESTRICTEDPYTHON_AVAILABLE
    if SANDBOX_MODULE_AVAILABLE:
        print("INFO: Sandbox module loaded successfully.")
except ImportError:
    SANDBOX_MODULE_AVAILABLE = False
    print("WARNING: Sandbox module not found. Sandboxed execution will be disabled.")
except Exception as e:
    SANDBOX_MODULE_AVAILABLE = False
    print(f"ERROR: Failed to load sandbox module: {e}")


'''     ==== ==== Proc section === ===     '''

def replace_envs(params):
    def replace_env_var(value):
        if isinstance(value, str) and value.startswith("ENV_"):
            env_var_name = value[4:]  # Remove "ENV_" prefix
            return os.environ.get(env_var_name, value)  # Return original value if env var not found
        return value

    # Create a deep copy of the parameters
    updated_params = copy.deepcopy(params)
    
    # Apply replacement only to top-level values in the dictionary
    for key, value in updated_params.items():
        updated_params[key] = replace_env_var(value)
    
    return updated_params
def needs_updated(params):
    for value in params.values():
        if isinstance(value, str) and value.startswith("ENV_"):
            return True
    return False

def exec_proc_agent(function_name: str, version_suffix: str, step_params: Dict[str, Any], function_def: str, 
        force_recompile: bool = False, jail_config: Dict[str, Any] = None
        )-> tuple[bytes, Dict[str, Dict[str, Union[int, str]]]]:

    spacing = depth_manager.get_spacing()
    logging.info("%sStarting %s%s" % (spacing, function_name, version_suffix))
    logging.debug("%s******** \n step_params%s" % (spacing, step_params))

    result = b''
    status = {"status": {"value": 1, "reason": "Function execution not attempted"}}
    
    try:
        updated_step_params = replace_envs(step_params) if needs_updated(step_params) else step_params
        if jail_config is None:
            #version the cached function name to stop collesions with other agents of the same name
            versioned_function_name = function_name + version_suffix
            function_def = re.sub(r'def\s+' + function_name, f'def {versioned_function_name}', function_def, 1)
            function_name = versioned_function_name
            if force_recompile or function_name not in _proc_agent_namespace:
                if force_recompile:
                    logging.info("%sForce recompile requested for function %s" % (spacing, function_name))
                else:
                    logging.info("%sCreating function %s for the first time" % (spacing, function_name))
                exec(function_def, _proc_agent_namespace)
            else:
                logging.info("%sUsing existing cached function %s from proc namespace" % (spacing, function_name))
            func = _proc_agent_namespace[function_name]
            result, status = func(**updated_step_params)
        else:
            logging.info("%sJailing function %s" % (spacing, function_name))
            result, status = sandbox_module._execute_jailed(function_name, function_def, updated_step_params, jail_config)    
    except Exception as e:
        logging.error("%sAn error occurred while executing %s: %s" % (spacing, function_name, str(e)))
        status = {"status": {"value": 1, "reason": "Error executing %s: %s" % (function_name, str(e))}}
    finally:
        if force_recompile and function_name in _proc_agent_namespace:
            logging.info("%sCleaning up temporary function '%s' from proc namespace." % (spacing, function_name))
            del _proc_agent_namespace[function_name]

    logging.info("%sCompleted %s with status: %s" % (spacing, function_name, str(status)))
    return result, status

'''   ==== ==== Workflow section === ===   '''
'''      == handle scoped variables ==     '''
def resolve_value(value: Any, scoped_params: Dict[str, Any]) -> Any:
    if isinstance(value, str) and value.startswith('$'):
        key =  resolve_value (value[1:],scoped_params)
        return scoped_params.get(key, value)
    return value
def add_to_scoped_params(scoped_params: Dict[str, Any], source: Dict[str, Any]) -> None:
    for key, value in source.items():
        #logging.debug(f" * added *   <-- key >{key}< >value >{value}<")
        scoped_params[key] = resolve_value(value, scoped_params)
def build_scoped_params(step_params: Dict[str, Any], cli_args: Dict[str, Any], results: Dict[str, Any]) -> Dict[str, Any]:
    scoped_params = {}
    # Add parameters in order of scope (closest to furthest)
    add_to_scoped_params(scoped_params, results)      # farthest
    add_to_scoped_params(scoped_params, cli_args)
    add_to_scoped_params(scoped_params, step_params)  # closest
    #logging.debug(f" * build scope params >{scoped_params}<")
    return scoped_params

''' == handle templates == '''
def process_step_params(step_params: Dict[str, Any], scoped_params: Dict[str, Any]) -> Dict[str, Any]:
    processed_params = {}
    for key, value in step_params.items():
        try:
            processed_params[key] = resolve_value(value, scoped_params)
        except ValueError as e:
            raise ValueError(f"Error processing parameter '{key}': {str(e)}")
    return processed_params
def build_template (prompt_template, scoped_params):
    if prompt_template: prompt = process_agent_params(prompt_template, scoped_params)
    else: prompt = ''   # report error, template class should have prompt.
    return prompt,{"status": {"value": 0, "reason": "Success"}}
def clean_json_string(s: str) -> str:
    """Clean a string to make it safe for insertion into JSON."""
    s = s.replace('\\', '\\\\').replace('"', '\\"')
    # Replace newlines with spaces
    s = s.replace('\n', '\\n').replace('\r', '\\n')
    # Escape backslashes and double quotes
    s = s.replace('\\\\\\\\\\\\', '\\\\')
    s = s.replace('\\\\\\\\\\', '\\\\')
    s = s.replace('\\\\\\\\', '\\\\')
    s = s.replace('\\\\\\', '\\\\')
    s = s.replace('\\\\', '\\')
    # Remove control characters
    s = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', s)
    return s
def process_agent_params(prompt: str, scoped_params: Dict[str, Any]) -> str:
    for key, value in scoped_params.items():
        placeholder = f"{{{key}}}"
        if placeholder in prompt:
            resolved_value = resolve_value(f"${key}", scoped_params)
            if resolved_value is not None:
                if isinstance(resolved_value, str):
                    cleaned_value = clean_json_string(resolved_value)
                elif isinstance(resolved_value, bytes):
                    cleaned_value = resolved_value.decode(errors='ignore')
                else:
                    cleaned_value = str(resolved_value)
                prompt = prompt.replace(placeholder, cleaned_value)
    return prompt

''' == handle nested workflows == ''' 
def filter_large_data(result):
    # Return a placeholder message for large data blocks
    trun_size= int(log_text_limit // 2 - 2) 
    if len(str(result)) > log_text_limit:
        truncated_result = f"{str(result)[:trun_size]}...{str(result)[-trun_size:]}"
        return truncated_result
    else:
        return result  
def get_nested_args(step, scoped_params, spacing, agent_name, config, cli_args):
    nested_cli_args = {}
    for param_name, param_value in step['params'].items():
        resolved_value = resolve_value(param_value, scoped_params)
        nested_cli_args[param_name] = resolved_value
        if resolved_value == param_value and isinstance(param_value, str) and param_value.startswith('$'):
            logging.warning(f"{spacing}Unable to resolve parameter '{param_name}' with value '{param_value}' in nested workflow '{agent_name}'")
    if not nested_cli_args:
        logging.warning(f"{spacing}No parameters could be resolved for nested workflow '{agent_name}'. This may indicate a mapping error.")
    # if dryrun exists, transfer it to nested_cli_args
    if "dryrun" in cli_args: 
        nested_cli_args["dryrun"] = cli_args["dryrun"]

    nested_workflow_name = step['agent']
    nested_workflow_config = config['agents'][nested_workflow_name].copy()  # Create a shallow copy of the workflow config
    
    # Replace the nested workflow's outputs with the current step's output
    nested_workflow_config['outputs'] = step['output'] if isinstance(step['output'], list) else [step['output']]
    
    logging.debug(f"{spacing}Nested workflow '{nested_workflow_name}' outputs set to: {nested_workflow_config['outputs']}")
    
    return nested_cli_args, nested_workflow_config

''' == handle depth == '''
from contextlib import contextmanager
class WorkflowDepthManager:
    def __init__(self, max_depth, indent_char=' .'):
        self.current_depth = 0
        self.max_depth = max_depth
        self.indent_char = indent_char

    @contextmanager
    def step(self):
        if self.current_depth >= self.max_depth:
            raise RuntimeError(f"Maximum Workflow Depth Exceeded: {self.current_depth}")
        self.current_depth += 1
        try:
            spacing = self.indent_char * (self.current_depth - 1) + ' '
            yield self.current_depth, spacing
        finally:
            self.current_depth -= 1

    def get_spacing(self):
        return self.indent_char * (self.current_depth - 1) + ' '

''' == handle workflow tasks == '''
def prepare_for_steps(workflow, results):
    steps = workflow['steps']
    results['step_index'] = 0  # Initialize step_index in results to give access to steps to control which step to execute next
    results['step_max'] = len(steps)
    return steps

def process_vars (steps, config, cli_args, results):
    step = steps[results['step_index']]
    full_agent_name = step['agent']
    agent_config = config['agents'][full_agent_name]
    
    # Extract version suffix and clean agent name
    version_match = re.search(r'^(.*)/([^/]+)/v(\d+)\.(\d+)$', full_agent_name)
    if version_match:
        clean_agent_name = version_match.group(2)
        major = version_match.group(3)
        minor = version_match.group(4)
        version_suffix = f"_v{major}_{minor}"
    else:
        clean_agent_name = full_agent_name
        version_suffix = ""

    # Build scoped parameters
    scoped_params = build_scoped_params(step.get('params', {}), cli_args, results)
    # Add step_index to scoped_params
    scoped_params['step_index'] = results['step_index']
    # Process step parameters
    step_params = process_step_params(step.get('params', {}), scoped_params)
    # Add the output parameter to step_params
    # so they can be seen in the proc to know what names to map when there are multiple results from one step.
    if 'output' in step:
        step_params['output'] = step['output']
        
    return step, clean_agent_name, agent_config, scoped_params, step_params, version_suffix

def handle_results(results, result, outvar):
    spacing = depth_manager.get_spacing()
    #pdb.set_trace()
    
    # If result is a dictionary, merge it with the existing results dictionary
    if isinstance(result, dict): 
        keys_to_store = list(result.keys())
        results.update(result)
        logging.debug(f"{spacing}Storing these keys: >>>{', '.join(keys_to_store)}<<<")
    # else store it with the output key
    else: 
        results[outvar] = result
        logging.debug(f"{spacing}Result stored as key >{outvar}<: >>>{filter_large_data(result)}<<<")

def update_status_step_index(results, status, return_on_fail, agent_name, duration):
    spacing = depth_manager.get_spacing()
    # if the status is a fail
    if status['status']['value'] == 1:  # Step failed
        logging.info(f"{spacing}Step Failed.      '{agent_name}' reason: {status['status']['reason']} in {format_time_interval(duration)}")
        if return_on_fail == 1:
            return True
    else: logging.info(f"{spacing}Step completed.      '{agent_name}' in {format_time_interval(duration)}")
    results.update(status)
    results['step_index'] = results['step_index'] +1
    return False
def get_final_results(results, result, outputs):
    spacing = depth_manager.get_spacing()
    final_results = {}
    
    for output in outputs:
        if output in results:
            value = results[output]
            if isinstance(value, str):
                try:
                    # Try to parse as JSON and then re-stringify to ensure proper escaping
                    final_results[output] = json.dumps(json.loads(value))
                except json.JSONDecodeError:
                    # If it's not valid JSON, store it as is
                    final_results[output] = value
            else:
                final_results[output] = value
        else:
            logging.warning(f"{spacing}Designated output '{output}' not found.")
            logging.debug(f"{spacing}Available outputs were: {', '.join(results.keys())}")
    
    if not final_results:
        logging.warning(f"{spacing}No designated outputs found. Returning last result.")
        if isinstance(result, dict):
            final_results = result
        else:
            final_results = {outputs[0]: result} if outputs else {"result": result}
    
    return final_results
def format_time_interval(elapsed_time):
    if elapsed_time >= 3600:  # 1 hour
        formatted_time = f"{elapsed_time / 3600:.4f} hours"
    elif elapsed_time >= 60:  # 1 minute
        formatted_time = f"{elapsed_time / 60:.4f} minutes"
    elif elapsed_time >= 1:  # 1 second
        formatted_time = f"{elapsed_time:.4f} seconds"
    elif elapsed_time >= 0.001:  # 1 millisecond
        formatted_time = f"{elapsed_time * 1000:.4f} milliseconds"
    else:  # microseconds
        formatted_time = f"{elapsed_time * 1000000:.4f} microseconds"
    return formatted_time

def validate_workflow(workflow: Dict[str, Any], config: Dict[str, Any]):
    # Check if the in-memory copy has already been blessed as valid.
    if workflow.get("_is_validated", False):
        return  # Skip validation entirely.

    spacing = depth_manager.get_spacing()
    logging.info(f"{spacing}Starting workflow validation")
    errors = []
    warnings = []
    
    outputs = set(["step_index", "status"])
    defined_params = set(workflow.get('inputs', []) + workflow.get('optional_inputs', []))

    for i, step in enumerate(workflow['steps']):
        agent_name = step['agent']
        
        if agent_name not in config['agents']:
            errors.append(f"{spacing}Step {i+1}: Agent '{agent_name}' is not defined in the configuration.")
            continue

        agent_config = config['agents'][agent_name]
        
        # Validate inputs
        for required_input in agent_config.get('inputs', []):
            if not any(required_input == param.strip() for param in step['params']):
                errors.append(f"{spacing}Step {i+1} ({agent_name}): Required input '{required_input}' is missing.")

        for input_param, value in step['params'].items():
            input_param = input_param.strip()
            if input_param not in agent_config.get('inputs', []) + agent_config.get('optional_inputs', []):
                warnings.append(f"{spacing}Step {i+1} ({agent_name}): Input '{input_param}' is not defined in the agent configuration.")
            
            if isinstance(value, str) and value.startswith('$'):
                param_name = value[1:]
                if param_name not in defined_params and param_name not in outputs:
                    warnings.append(f"{spacing}Step {i+1} ({agent_name}): Input '{input_param}' uses '{value}' which is not an output from any previous step or a defined input.")

        # Validate output existence (but not its specific value)
        if 'output' not in step:
            errors.append(f"{spacing}Step {i+1} ({agent_name}): Missing 'output' definition.")
        else:
            step_outputs = step['output'] if isinstance(step['output'], list) else [step['output']]
            outputs.update(step_outputs)

        # Validate agent type
        agent_type = agent_config.get('type')
        if agent_type not in ['template', 'proc', 'workflow', 'step']:
            errors.append(f"{spacing}Step {i+1} ({agent_name}): Invalid agent type '{agent_type}'.")

        # Specific checks for non-workflow agents
        if agent_type == 'template' and 'prompt' not in agent_config:
            errors.append(f"{spacing}Step {i+1} ({agent_name}): Template agent missing 'prompt' definition.")
        elif agent_type == 'proc' and ('function' not in agent_config or 'function_def' not in agent_config):
            errors.append(f"{spacing}Step {i+1} ({agent_name}): Proc agent missing 'function' or 'function_def'.")

    if errors:
        for error in errors:
            logging.error(error)
        raise ValueError("Workflow validation failed. Please check the errors above.")

    if warnings:
        for warning in warnings:
            logging.warning(warning)

    # If we reach here, validation was successful. Bless the in-memory copy.
    workflow["_is_validated"] = True
    logging.info(f"{spacing}Workflow validation completed successfully and has been blessed.")

def exec_workflow(workflow: Dict[str, Any], config: Dict[str, Any], cli_args: Dict[str, Any],results,
                  force_recompile: bool = False, jail_config: Dict[str, Any] = None)->bytes:
    with depth_manager.step() as (depth, spacing):
    
        logging.info(f"{spacing}Executing workflow at depth {depth}")
        logging.debug(f"{spacing}cli_args: {cli_args}")

        try:
            validate_workflow(workflow, config)
        except ValueError as e:
            logging.error(f"{spacing}Workflow validation failed: {str(e)}")
            raise

        steps = prepare_for_steps(workflow, results)
        while results['step_index'] < len(steps):
            start_time = time.perf_counter()
            step, agent_name, agent_config, scoped_params, step_params, version_suffix = process_vars(steps, config, cli_args, results)
            logging.info(f"{spacing}Executing step: {agent_name}, ver. {version_suffix}  type : {agent_config['type']}")
 
            try:
                if agent_config['type'] == 'template':
                    result, status = build_template(agent_config.get('prompt', ''), scoped_params)
                elif agent_config['type'] == 'proc':
                    result, status = exec_proc_agent(agent_config['function'], version_suffix, step_params, agent_config['function_def'], 
                                    force_recompile, jail_config)  
                elif agent_config['type'] == 'workflow':
                    nested_cli_args, nested_workflow = get_nested_args (step, scoped_params, spacing, agent_name, config, cli_args)
                    result, status = exec_workflow(nested_workflow, config, nested_cli_args, {}, 
                                    force_recompile, jail_config)
                else:  #unknown agent
                    result = b''
                    status = {"status": {"value": 1, "reason": f"Unknown agent type: {agent_config['type']}"}}

                handle_results(results, result, step['output'][0])

            except Exception as e:
                error_msg = f"Error executing step '{agent_name}': {str(e)}"
                error_context = f"Depth: {depth}, Previous steps: {results['step_index']}, Params: {step_params}"
                logging.error(f"{spacing}{error_msg}\nContext: {error_context}")
                status = {"status": {"value": 1, "reason": error_msg, "context": error_context}}

            if update_status_step_index(results, status, workflow.get('return_on_fail', 0), agent_name, (time.perf_counter() - start_time)):
                return b'', status
        
        return get_final_results(results, result, workflow['outputs']), {"status": {"value": 0, "reason": "Success"}}

def exec_agent(agent: Dict[str, Any], agent_name: str, config: Dict[str, Any], cli_args: Dict[str, Any],results, 
                force_recompile: bool = False, jail_config: Dict[str, Any] = None)->bytes:
    # If the agent has no steps, promote it to a temporary workflow
    if not agent.get('type') in ['workflow']:
        agent = create_temp_workflow(agent_name, agent, cli_args)
    return  exec_workflow(agent, config, cli_args, results, force_recompile, jail_config)

'''    ==== ==== main setup section === ===    '''
def setup_logging(verbose_level, log_server=None):
    root_logger = logging.getLogger()
    # Clear existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    if verbose_level == 0:
        root_logger.setLevel(logging.WARNING)
    elif verbose_level == 1:
        root_logger.setLevel(logging.INFO)
    else:
        root_logger.setLevel(logging.DEBUG)

    # Console handler
    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s - %(funcName)s - %(lineno)d')
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)
    # Define the logging format

    # Remote logging
    if log_server:
        try:
            host, port = log_server.split(':')
            port = int(port)
            socket_handler = logging.handlers.SocketHandler(host, port)
            root_logger.addHandler(socket_handler)
            logging.info(f"Remote logging enabled to {host}:{port}")
        except Exception as e:
            logging.error(f"Failed to set up remote logging: {e}")
def create_temp_workflow(agent_name, agent_config, cli_args):
    logging.info(f"Creating temporary workflow agent for agent: {agent_name}")
    #logging.debug(f"Agent config: {agent_config}")

    # creating a temp workflow to promote an agent that is not a workflow to be a workflow
    temp_workflow = {'type': 'workflow', 'help': 'A singleton entry to allow agents to execute.', 'return_on_fail': 1, 'inputs': [], 'optional_inputs': [], 'outputs': ['image'], 'prompt': '{prompt}', 'steps': [{'agent': '{agent}', 'host': 'stable_diffusion', 'model': '', 'api_call': 'prompting', 'params': {'topic': '$CLI_topic', 'thesis': '$CLI_thesis', 'essay': '$fact_checked_essay', 'tone': '$CLI_tone'}, 'output': ['results']}]}

    temp_workflow['steps'][0]['agent'] = agent_name
    temp_workflow['steps'][0]['params'] = {input_name: f"${input_name}" for input_name in agent_config['inputs']}
    temp_workflow['steps'][0]['output'] = agent_config['outputs']
    temp_workflow['inputs'] = agent_config['inputs']
    temp_workflow['optional_inputs'] = agent_config.get('optional_inputs', [])
    temp_workflow['outputs'] = agent_config['outputs']
    temp_workflow['help'] = agent_config['help']
    for opt_input in agent_config.get('optional_inputs', []):
        if cli_args.get(opt_input) is not None:
            # Add to workflow inputs
            temp_workflow['inputs'].append(opt_input)
            # Add to steps params
            temp_workflow['steps'][0]['params'][opt_input] = f"${opt_input}"
    #print (temp_workflow)
    return temp_workflow
def load_config(default_file_path: str) -> Dict[str, Any]:
    logging.info(f"Starting config load")
    
    # First stage: Create a minimal parser just for the config file path
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument('--config', default=default_file_path, 
                             help='Path to configuration file')
    if SANDBOX_MODULE_AVAILABLE : 
        sandbox_module.add_sandbox_args(config_parser)
    
    # Parse only known args to get the config path, ignore everything else
    config_args, _ = config_parser.parse_known_args()
    config_file_path = config_args.config
    
    # Load the JSON config file
    try:
        with open(config_file_path, 'r') as f:
            return json.load(f)
        
    except FileNotFoundError:
        print(f"Error: Configuration file '{config_file_path}' not found.")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Configuration file '{config_file_path}' is not a valid JSON.")
        sys.exit(1)

def config_app():
    config = load_config('config.json')
    
    # Second stage: Create the full parser with all options
    parser = argparse.ArgumentParser(description="Dynamic Agent Workflow Script")
    parser.add_argument('agent', nargs='?', help='Name of the agent to execute')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    parser.add_argument('-v', '--verbose', action='count', default=0, 
                       help='Increase output verbosity (e.g., -v, -vv, -vvv)')
    parser.add_argument('--log-server', help='Enable remote logging to server:port')
    parser.add_argument('--dryrun', action='store_true', 
                       help='Bypasses network call, returns dummy message')
    if SANDBOX_MODULE_AVAILABLE : 
        sandbox_module.add_sandbox_args(parser)

    # If no agent specified, show available agents
    if len(sys.argv) == 1 or (len(sys.argv) == 3 and '--config' in sys.argv):
        print("Available agents:")
        for name, agent in config['agents'].items():
            if name != 'singleton':  # Exclude singleton from the list
                print(f"  {name}: {agent['help']}")
                print(f"    Inputs: {', '.join(agent['inputs'])}")
                if agent.get('optional_inputs'):
                    print(f"    Optional inputs: {', '.join(agent['optional_inputs'])}")
                print(f"    Outputs: {', '.join(agent['outputs'])}")
        sys.exit()

    # Parse all arguments to get the agent name first
    temp_args, _ = parser.parse_known_args()
    
    if not temp_args.agent:
        print("Available agents:")
        for name, agent in config['agents'].items():
            if name != 'singleton':
                print(f"  {name}: {agent['help']}")
                print(f"    Inputs: {', '.join(agent['inputs'])}")
                if agent.get('optional_inputs'):
                    print(f"    Optional inputs: {', '.join(agent['optional_inputs'])}")
                print(f"    Outputs: {', '.join(agent['outputs'])}")
        sys.exit()
    
    agent_config = config['agents'].get(temp_args.agent)
    if not agent_config:
        print(f"Error: '{temp_args.agent}' is not a valid agent.")
        sys.exit()

    # Third stage: Create final parser with agent-specific arguments
    final_parser = argparse.ArgumentParser(description=agent_config['help'])
    final_parser.add_argument('agent', help='Name of agent to execute')
    final_parser.add_argument('--config', default='config.json', help='Path to configuration file')
    
    # Add agent-specific required inputs
    for input_name in agent_config['inputs']:
        final_parser.add_argument(f'--{input_name}', required=True, 
                                help=f'Value for {input_name}')
    
    # Add agent-specific optional inputs
    for input_name in agent_config.get('optional_inputs', []):
        final_parser.add_argument(f'--{input_name}', 
                                help=f'Value for {input_name}')
    
    # Add common arguments
    final_parser.add_argument('-v', '--verbose', action='count', default=0, 
                            help='Increase output verbosity (e.g., -v, -vv, -vvv)')
    final_parser.add_argument('--log-server', help='Enable remote logging to server:port')
    final_parser.add_argument('--dryrun', action='store_true', 
                            help='Bypasses network call, returns dummy message')
    if SANDBOX_MODULE_AVAILABLE : 
        sandbox_module.add_sandbox_args(final_parser)
    
    # Final parse with all arguments
    args = final_parser.parse_args()
    setup_logging(args.verbose, args.log_server)

    # Only add actual entries that appeared on the command line
    cli_args = {k: v for k, v in vars(args).items() if v is not None}

    return agent_config, config, cli_args, args.agent, args
def setup_depth_manager(config):
    global depth_manager

     # Ensure the workflow_settings exist and have a default max_depth
    if 'workflow_settings' not in config:
        config['workflow_settings'] = {}
    if 'max_depth' not in config['workflow_settings']:
        config['workflow_settings']['max_depth'] = 20  # Default value

    max_depth = config['workflow_settings']['max_depth']
    depth_manager = WorkflowDepthManager(max_depth=max_depth)

def main():
    start_time = time.perf_counter()
    #pdb.set_trace()
    agent, config, cli_args, agent_name, args = config_app()
    setup_depth_manager(config)
    global log_text_limit
    log_text_limit= int(config['workflow_settings']['log_text_limit'])
    results = {}
    jail_config = sandbox_module.setup_jail_config(args) if SANDBOX_MODULE_AVAILABLE else None

    try:
        logging.info(f"Starting execution of workflow: {agent_name}")
        result, status = exec_agent(agent, agent_name, config, cli_args, results, jail_config=jail_config)
        if isinstance(result, dict):
            for key, value in result.items():
                if isinstance(value, bytes):
                    sys.stdout.buffer.write(f"{key}: ".encode())
                    sys.stdout.buffer.write(value)
                    sys.stdout.buffer.write(b"\n")
                else:
                    print(f"{key}: {value}")
        elif isinstance(result, bytes):
            sys.stdout.buffer.write(result)
        else:
            print(result)

        if status['status']['value'] == 0:
            logging.info(f"Workflow '{agent_name}' completed successfully. Reason: {status['status']['reason']} in {format_time_interval(time.perf_counter() - start_time)}")
        else:
            logging.error(f"Workflow '{agent_name}' failed. Reason: {status['status']['reason']} in {format_time_interval(time.perf_counter() - start_time)}")
        
        # If you want to print additional details from the status:
        if 'context' in status['status']:
            logging.info(f"Additional context: {status['status']['context']}")

    except ValueError as e: 
        logging.error(f"Error in workflow configuration: {str(e)}")
    except Exception as e: 
        logging.error(f"An error occurred during workflow execution: {str(e)}") 
    
    sys.exit(status['status']['value'])

if __name__ == "__main__":
    main()
