#!/home/jrogers/Documents/ai/chat/ragserver/venv/bin/python3

from html.parser import HTMLParser
import argparse
import json
import urllib.parse
from typing import Union, Dict, Any
import logging
import logging.handlers
import socket
import requests
import sys
import re
import lxml as etree
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import base64
import pdb
import time

''' ==== ==== Proc testing section === === '''

def get_length_of_list(list_input: any, output: list) -> tuple[Dict, Dict]:
    status = {"status": {'value': 0, 'reason': 'Success'}}
    
    if isinstance(list_input, list):
        list_len = len(list_input)
    else:
        list_len = -1
        status = {"status": {'value': 1, 'reason': 'Not a list'}}

    return list_len, status


import ast
def convert_json_to_results3(content: Union[str, dict], output: list) -> tuple[Dict, Dict]:
    status = {"status": {'value': 0, 'reason': 'Success'}}
    pdb.set_trace()
    if isinstance(content, str):
        try:
            # Remove any leading/trailing whitespace
            content = content.strip()
            content = content.replace("\\\\\\\"", "'")
            
            # Try to convert the content string to a dictionary using json.loads
            content = json.loads(content)
        except json.JSONDecodeError:
            try:
                # If json.loads fails, try using ast.literal_eval
                content = ast.literal_eval(content)
            except (ValueError, SyntaxError):
                # If both methods fail, set the content to an empty dict and update the status
                content = {}
                status = {"status": {'value': 1, 'reason': 'Failed to convert content to json'}}
            
    pdb.set_trace()
    return content, status

from hugchat import hugchat
from hugchat.login import Login
def hugging_text_completion(hugging_request:str,hugging_model:int, hugging_user_name:str, hugging_password:str, output:list)->str:
    #rate limiting
    time.sleep(1)
    # Log in to huggingface and grant authorization to huggingchat
    cookie_path_dir = "./cookies/" # NOTE: trailing slash (/) is required to avoid errors
    sign = Login(hugging_user_name, hugging_password)
    cookies = sign.login(cookie_dir_path=cookie_path_dir, save_cookies=True)

    chatbot = hugchat.ChatBot(cookies=cookies.get_dict(), default_llm=hugging_model)  # or cookie_path="usercookies/<email>.json"
    #message_generator = chatbot.chat(hugging_request) # note: is a generator, the method will return immediately.
    message_result = chatbot.chat(hugging_request) # note: is a generator, the method will return immediately.

    # Collect results from the generator
    #message_result: str = message_generator.wait_until_done() 
    
    return {output[0]: message_result}, {"status": {"value": 0, "reason": "Success"}}

from groq import Groq
def groq_text_completion(
    groq_request: str,
    groq_model: str,
    groq_api_key: str,
    output: list,
    max_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 1.0,
    n: int = 1,
    stop: list = None,
    rate_limit: int = 60
) -> str:
    time.sleep(rate_limit)
    client = Groq(
        api_key=groq_api_key,
    )

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": groq_request,
            }
        ],
        model=groq_model,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=n,
        stop=stop,
    )

    return {output[0]: chat_completion.choices[0].message.content}, {"status": {"value": 0, "reason": "Success"}}

import openai
def chatgpt_text_completion(
    chatgpt_request: str,
    chatgpt_model: str,
    openai_api_key: str,
    output: list,
    temperature: float = 0.7,
    max_tokens: int = 150,
    top_p: float = 1.0,
    frequency_penalty: float = 0.0,
    presence_penalty: float = 0.0,
    rate_limit: int = 60
) -> str:
    # Rate limiting
    time.sleep(rate_limit)

    # Set the OpenAI API key
    openai.api_key = openai_api_key

    # Generate a response using the ChatGPT API
    response = openai.ChatCompletion.create(
        model=chatgpt_model,
        messages=[
            {"role": "user", "content": chatgpt_request}
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        frequency_penalty=frequency_penalty,
        presence_penalty=presence_penalty
    )

    # Extract the response message
    message_result = response['choices'][0]['message']['content']

    return {output[0]: message_result}, {"status": {"value": 0, "reason": "Success"}}



def emit_part_by_index_offset(parts: list, index: int, offset:int, out_of_range_part:str, output:list) -> str:
		if index+offset >= len(parts):
			return {output[0]: out_of_range_part}, {"status": {"value": 0, "reason": "Success, emitted alternative item"}}
		return {output[0]: parts[index+offset]}, {"status": {"value": 0, "reason": "Success"}}

def extract_json_trim(data: bytes, trim: int, output: list) -> bytes:
    length = len(data)
    #pdb.set_trace()
    return {output[0]: data}, {"status": {"value": 0, "reason": "Success"}}
    #return {output[0]: data[0:length-trim]}, {"status": {"value": 0, "reason": "Success"}}

import json_repair
def extract_json(data: bytes, output: list) -> bytes:
    # Find the first occurrence of '{'
    try:
        data = data.replace('\\n', '\n')
        data = re.sub(r'}\s*{', '},{', data)
        start_index = None
        #pdb.set_trace()
        for i, char in enumerate(data):
            if char == '{':
                start_index = i
                break

        if start_index is None:
            return data, {"status": {"value": 1, "reason": "Failed to find start tag"}}

        # Find the last occurrence of '}'
        end_index = None
        for i in range(len(data) - 1, start_index, -1):
            if data[i] == '}':
                end_index = i
                break

        if end_index is None:
            return data, {"status": {"value": 1, "reason": "Failed to find end tag"}}
        result = data[start_index:end_index + 1]

        fixed_result = repair_json(result)
        parsed_result = json.loads(fixed_result)

        # If output[0] key is present in parsed_result, no need to add it again
        return parsed_result, {"status": {"value": 0, "reason": "Success"}}

    except Exception as e:
        return data, {"status": {"value": 1, "reason": str(e)}}

def debug_print(input: str, output:list) -> bytes:
		print (f"*** >>>{input}<<<")
		return {output[0]: b''}, {"status": {"value": 0, "reason": "Success"}}

import urllib.parse
def build_url_request(protocol:str, host:str, endpoint: str, request: str, output:list)->str: 
    # f"https://www.googleapis.com/customsearch/v1?q={query}"
    fixed_request= urllib.parse.quote_plus(request)
    return f"{protocol}://{host}/{endpoint}{fixed_request}", {"status": {"value": 0, "reason": "Success"}}


#        "build_google_url_header_request": {
#            "type": "proc",
#            "help": "create url with request https://www.googleapis.com/customsearch/v1?q={query}",
#            "function": "build_url_request",
#            "function_def": "def needs work\n",
#            "inputs": [
#                "protocol", "host", "endpoint", "request","api_key", "cx"
#            ],
#            "optional_inputs": ["country_code", "dateRestrict_d", "dateRestrict_w", "dateRestrict_m",
#                "dateRestrict_y", "exactTerms", "excludeTerms", "fileType", "geolocation", "imgColorType",
#                "imgDominantColor", "imgSize", "imgType", "linkSite", "language", "lowRange", "highRange",
#                "num", "orTerms", "rights", "safe", "searchType", "siteSearch", "sort",
#                "siteSearchFilterInclude", "siteSearchFilterExclude", "sort", "start" ],
#            "outputs": [
#                "url", "headers"
#            ]
#        }


import urllib.parse
import inspect
def build_google_url_header_request (request: str, api_key:str, cx: str, output:list,
    country_code: str = "", dateRestrict_d: str = "", dateRestrict_w: str = "", dateRestrict_m: str = "",
    dateRestrict_y: str = "", exactTerms: str = "", excludeTerms: str = "", fileType: str = "", geolocation: str = "",
    imgColorType: str = "",imgDominantColor: str = "", imgSize: str = "", imgType: str = "", linkSite: str = "", 
    language: str = "", lowRange: str = "", highRange: str = "", num: str = "", orTerms: str = "", rights: str = "",
    safe: str = "", searchType: str = "", siteSearch: str = "", sort: str = "", siteSearchFilterInclude: str = "",
    siteSearchFilterExclude: str = "", start: str = "") -> tuple[Dict, Dict]:

    headers = (f"{{\"Content-Type\": \"application/json\", \"Accept\": \"application/json\", "
          f"\"User-Agent\": \"My Custom Search Client/1.0\", \"X-Goog-Api-Key\": \"{api_key}\"}}")

    fixed_request= urllib.parse.quote(request)

    # f"https://www.googleapis.com/customsearch/v1?q={query}"
    url = f"https://www.googleapis.com/customsearch/v1?q={fixed_request}&cx={cx}"

    all_params = build_google_url_header_request.__code__.co_varnames[:build_google_url_header_request.__code__.co_argcount]
    num_required = build_google_url_header_request.__code__.co_argcount - len(build_google_url_header_request.__defaults__)
    optional_params = all_params[num_required:]

    # Add optional parameters to URL if they're not empty
    for param in optional_params:
        value = locals()[param]
        if value != "":
            url += f"&{param}={urllib.parse.quote(str(value))}"  

    return { "url": url, "headers": headers } , {"status": {"value": 0, "reason": "Success"}}

'''     ==== ==== Proc section === ===     '''
import os
import copy
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
def exec_proc_agent(function_name: str, step_params: Dict[str, Any], function_def: str) -> tuple[bytes, Dict[str, Dict[str, Union[int, str]]]]:
    spacing = depth_manager.get_spacing()
    logging.info("%sStarting %s" % (spacing, function_name))
    logging.debug("%s******** \n step_params%s" % (spacing, step_params))
    
    result = b''  # Initialize result as empty bytes
    status = {"status": {"value": 1, "reason": "Function execution not attempted"}}
    
    try:
        # Dynamically create abd dispatch to the appropriate processing function
        # Check if the function is already defined in the global scope
        if function_name not in globals():
            logging.info("%sCreating function %s" % (spacing, function_name))
            logging.debug("%s******** Function definition:\n%s" % (spacing, function_def))
            # Define the function dynamically
            exec(function_def, globals())
        else: logging.info("%sUsing existing function %s" % (spacing, function_name))

        func = globals()[function_name]
        
        if needs_updated(step_params):
            updated_step_params = replace_envs(step_params)
        else:
            updated_step_params = step_params
        result, status = func(**updated_step_params)
        # return the status of the function executiong
        
    except Exception as e:
        logging.error("%sAn error occurred while executing %s: %s" % (spacing, function_name, str(e)))
        status = {"status": {"value": 1, "reason": "Error executing %s: %s" % (function_name, str(e))}}

    logging.info("%sCompleted %s with status: %s" % (spacing, function_name, str(status)))
    return result, status

'''   ==== ==== Workflow section === ===   '''
''' == handle scoped variables == '''
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
    agent_name = step['agent']
    agent_config = config['agents'][agent_name]
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
    return step, agent_name, agent_config, scoped_params, step_params
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
    spacing = depth_manager.get_spacing()
    final_results = {}
    
    for output in outputs:
        if output in results:
            final_results[output] = results[output]
        else:
            logging.warning(f"{spacing}Designated output '{output}' not found.")
            logging.debug(f"{spacing}Available outputs were: {', '.join(results.keys())}")
    
    if not final_results:
        final_results[outputs[0]] = result
    
    return final_results
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
            errors.append(f"Step {i+1} ({agent_name}): Missing 'output' definition.")
        else:
            step_outputs = step['output'] if isinstance(step['output'], list) else [step['output']]
            outputs.update(step_outputs)

        # Validate agent type
        agent_type = agent_config.get('type')
        if agent_type not in ['template', 'proc', 'workflow', 'step']:
            errors.append(f"Step {i+1} ({agent_name}): Invalid agent type '{agent_type}'.")

        # Specific checks for non-workflow agents
        if agent_type == 'template' and 'prompt' not in agent_config:
            errors.append(f"Step {i+1} ({agent_name}): Template agent missing 'prompt' definition.")
        elif agent_type == 'proc' and ('function' not in agent_config or 'function_def' not in agent_config):
            errors.append(f"Step {i+1} ({agent_name}): Proc agent missing 'function' or 'function_def'.")

        # For workflow agents, we don't perform additional output checks

    # We no longer check if all workflow outputs are produced by steps

    if errors:
        for error in errors:
            logging.error(error)
        raise ValueError("Workflow validation failed. Please check the errors above.")

    if warnings:
        for warning in warnings:
            logging.warning(warning)

    logging.info(f"{spacing}Workflow validation completed successfully.")
def exec_workflow(workflow: Dict[str, Any], config: Dict[str, Any], cli_args: Dict[str, Any],results)->bytes:
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
            step, agent_name, agent_config, scoped_params, step_params = process_vars(steps, config, cli_args, results)
            logging.info(f"{spacing}Executing step: {agent_name}, type : {agent_config['type']}")
 
            try:
                if agent_config['type'] == 'template':
                    result, status = build_template(agent_config.get('prompt', ''), scoped_params)
                elif agent_config['type'] == 'proc':
                    result, status = exec_proc_agent(agent_config['function'], step_params, agent_config['function_def'])  
                elif agent_config['type'] == 'workflow':
                    nested_cli_args, nested_workflow = get_nested_args (step, scoped_params, spacing, agent_name, config, cli_args)
                    result, status = exec_workflow(nested_workflow, config, nested_cli_args, {})
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
def create_temp_workflow(agent_name, agent_config, config):
    logging.info(f"Creating temporary workflow agent for agent: {agent_name}")
    logging.debug(f"Agent config: {agent_config}")

    singleton = config['agents']['singleton']
    temp_workflow = json.loads(json.dumps(singleton))  # Deep copy
    temp_workflow['steps'][0]['agent'] = agent_name
    temp_workflow['steps'][0]['params'] = {input_name: f"${input_name}" for input_name in agent_config['inputs']}
    temp_workflow['steps'][0]['output'] = agent_config['outputs']
    temp_workflow['inputs'] = agent_config['inputs']
    temp_workflow['optional_inputs'] = agent_config.get('optional_inputs', [])
    temp_workflow['outputs'] = agent_config['outputs']
    temp_workflow['help'] = agent_config['help']
    return temp_workflow
def load_config(default_file_path: str) -> Dict[str, Any]:
    logging.info(f"Starting : {format}")
    #logging.debug(f" {format, selector, response_text}")

    # Check if '--config' is in the command line arguments
    config_file_path = default_file_path
    if '--config' in sys.argv:
        # Get the index of '--config' and retrieve the next argument as the config file path
        config_index = sys.argv.index('--config') + 1
        if config_index < len(sys.argv):
            config_file_path = sys.argv[config_index]
            # Remove '--config' and the file path from sys.argv
            sys.argv.pop(config_index)  # Remove the file path
            sys.argv.pop(config_index - 1)  # Remove '--config'
        else:
            print("Error: --config option requires a file path argument.")
            sys.exit(1)
    
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
    
    parser = argparse.ArgumentParser(description="Dynamic Agent Workflow Scriptr")
    parser.add_argument('agent', nargs='?', help='Name of the agent to execute')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase output verbosity (e.g., -v, -vv, -vvv)')
    parser.add_argument('--log-server', help='Enable remote logging to server:port')
    parser.add_argument('--dryrun', action='store_true', help='Bypasses network call, returns dummy message')
    
    args, unknown = parser.parse_known_args()

    if not args.agent:
        print("Available agents:")
        for name, agent in config['agents'].items():
            if name != 'singleton':  # Exclude singleton from the list
                print(f"  {name}: {agent['help']}")
                print(f"    Inputs: {', '.join(agent['inputs'])}")
                if agent.get('optional_inputs'):
                    print(f"    Optional inputs: {', '.join(agent['optional_inputs'])}")
                print(f"    Outputs: {', '.join(agent['outputs'])}")
        sys.exit()
    
    if args.agent == 'singleton':
        print("Error: 'singleton' is not a valid agent.")
        sys.exit()

    agent_config = config['agents'].get(args.agent)
    if not agent_config:
        print(f"Error: '{args.agent}' is not a valid agent.")
        sys.exit()

    if not agent_config.get('type') in ['workflow']:
        # If the agent has no steps, promote it to a temporary workflow
        agent = create_temp_workflow(args.agent, agent_config, config)
    else:
        agent = agent_config

    parser = argparse.ArgumentParser(description=agent_config['help'])
    for input_name in agent_config['inputs']:
        parser.add_argument(f'--{input_name}', required=True, help=f'Value for {input_name}')
    for input_name in agent_config.get('optional_inputs', []):
        parser.add_argument(f'--{input_name}', help=f'Value for {input_name}')
    parser.add_argument('agent', help='Name of agent to execute')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='Increase output verbosity (e.g., -v, -vv, -vvv)')
    parser.add_argument('--log-server', help='Enable remote logging to server:port')
    parser.add_argument('--dryrun', action='store_true', help='Bypasses network call, returns dummy message')
    
    args = parser.parse_args()
    setup_logging(args.verbose, args.log_server)
    cli_args = vars(args)
    return agent, config, cli_args, args.agent
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
    agent, config, cli_args, agent_name = config_app()
    setup_depth_manager(config)
    global log_text_limit
    log_text_limit= int(config['workflow_settings']['log_text_limit'])
    results = {}

    try:
        logging.info(f"Starting execution of workflow: {agent_name}")
        result, status = exec_workflow(agent, config, cli_args, results)
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
