import sys
import re

def escape_json_string(input_str):
    output_bytes = bytearray()
    for char in input_str:
        if char == '\\':
            output_bytes.extend(b'\\\\')
        elif char == '"':
            output_bytes.extend(b'\\"')
        elif char == '/':
            output_bytes.extend(b'\\/')
        elif char == '\n':
            output_bytes.extend(b'\\n')
        elif char == '\r':
            output_bytes.extend(b'\\r')
        elif char == '\t':
            output_bytes.extend(b'\\t')
        else:
            encoded_char = char.encode('utf-8')
            output_bytes.append(ord(encoded_char))

    return output_bytes.decode('utf-8')

# Example usage
input_str = '''
def loop_start(step_index: int, loop_max: int, output:list) -> dict:
    return { output[0]: step_index,  output[1]: int(loop_max), output[2]: 0 }, {"status": {"value": 0, "reason": "Success"}}




def loop_end(step_index: int, loop_max: int, loop_index: int, output:list) -> any:
    new_counter = loop_index + 1
    if new_counter < loop_max:
        return {'step_index': step_index, output[0]: new_counter}, {"status": {"value": 0, "reason": "Success"}}
    return new_counter, {"status": {"value": 0, "reason": "Success"}}




def emit_part_by_index(parts: list, index: int,output:list) -> str:
        if index >= len(parts):
            raise IndexError("Index out of range")
        return {output[0]: parts[index]}, {"status": {"value": 0, "reason": "Success"}}




def add_item_to_list(item: Any, list_input: Union[str, list], output: list) -> tuple[list, Dict]:
    status = {"status": {'value': 0, 'reason': 'Success'}}
    if isinstance(list_input, list):
        content = list_input + [item]
    elif isinstance(list_input, str):
        content = [item]
        status = {"status": {'value': 0, 'reason': 'Success, converted item to a list'}}
    return content, status




def collect_item_to_list(item: Any, list_input: Union[str, list], output: list) -> tuple[list, Dict]:
    status = {"status": {'value': 0, 'reason': 'Success'}}
    if isinstance(list_input, list):
        list_input.append(item)
        content = { output[0]: list_input}
    else:
        # Create a new list and add the item to it
        content = { output[0]: [item]}
        status = {"status": {'value': 0, 'reason': 'Success, created new list with item'}}
    return content, status




def convert_json_to_results(content: Union[str, dict], output: list) -> tuple[Dict, Dict]:
    status = {"status": {'value': 0, 'reason': 'Success'}}

    if isinstance(content, str):
        try:
            # Try to convert the content string to a dictionary
            content = json.loads(content)
        except json.JSONDecodeError:
            # If the conversion fails, set the content to an empty dict and update the status
            content = b''
            status = {"status": {'value': 1, 'reason': 'Failed to convert content to json'}}

    return content, status




def get_length_of_list(list_input: any, output: list) -> tuple[Dict, Dict]:
    status = {"status": {'value': 0, 'reason': 'Success'}}

    if isinstance(list_input, list):
        list_len = len(list_input)
    else:
        list_len = -1
        status = {"status": {'value': 1, 'reason': 'Not a list'}}

    return list_len, status




def extract_json(data: bytes, output: list) -> bytes:
    # Find the first occurrence of '{'
    start_index = None
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

    return {output[0]: data[start_index:end_index + 1]}, {"status": {"value": 0, "reason": "Success"}}




import requests
import time
from requests.exceptions import RequestException, Timeout, ConnectionError
def exec_api_call(url: str, headers: Dict[str, str], payload: str, output:list) -> tuple[bytes, Dict[str, Dict[str, Union[int, str]]]]:
    max_retries: int = 3
    retry_delays = [60, 120, 240]  # 1 minute, 2 minutes, 4 minutes backoff
    retryable_status_codes = {408, 429, 500, 502, 503, 504}

    headers_dict = json.loads(headers) if isinstance(headers, str) else headers # Convert headers string to dictionary
    payload_bytes = payload.encode('utf-8', errors='replace') if isinstance(payload, str) else payload # Convert payload to bytes

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers_dict, data=payload_bytes,timeout=1200)
            #response = requests.post(url, headers=headers, data=payload.encode('latin-1'), timeout=30)
            
            if response.status_code == 200:
                # Decode response content, replacing any problematic characters
                response_text = response.content.decode('utf-8', errors='replace') 
                return response_text, {"status": {"value": 0, "reason": "Success"}}
            elif response.status_code in retryable_status_codes:
                if attempt < max_retries - 1:
                    delay = retry_delays[attempt]
                    logging.warning("Received status code %d. Retrying in %d seconds." % (response.status_code, delay))
                    time.sleep(delay)
                else:
                    return b"", {"status": {"value": 1, "reason": "%d HTTP Error after %d attempts: %s" % (response.status_code, max_retries, response.text)}}
            else:
                # Return the specific status code for non-retryable responses
                return b"", {"status": {"value": 1, "reason": "%d HTTP Error: %s" % (response.status_code, response.text)}}
                
        except Timeout as e:
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                logging.warning("API call timed out. Retrying in %d seconds. Error: %s" % (delay, str(e)))
                time.sleep(delay)
            else:
                logging.error("API call timed out after %d attempts. Error: %s" % (max_retries, str(e)))
                return b"", {"status": {"value": 1, "reason": "HTTP Error 408 API call timed out after %d attempts: %s" % (max_retries, str(e))}}
                
        except ConnectionError as e:
            if attempt < max_retries - 1:
                delay = retry_delays[attempt]
                logging.warning("Connection error. Retrying in %d seconds. Error: %s" % (delay, str(e)))
                time.sleep(delay)
            else:
                logging.error("Connection failed after %d attempts. Error: %s" % (max_retries, str(e)))
                return b"", {"status": {"value": 1, "reason": "HTTP Error 503 Connection failed after %d attempts: %s" % (max_retries, str(e))}}
                
        except RequestException as e:
            # For other request exceptions, don't retry
            logging.error("Request exception: %s" % str(e))
            return b"", {"status": {"value": 1, "reason": "HTTP Error 400 Request exception: %s" % str(e)}}
            
        except Exception as e:
            # For other unexpected exceptions, don't retry
            logging.error("Unexpected error in API call: %s" % str(e))
            return b"", {"status": {"value": 1, "reason": "HTTP Error 500 Unexpected error in API call: %s" % str(e)}}
    
    return b"", {"status": {"value": 1, "reason": "HTTP Error 429 Max retries reached without successful API call"}}
'''

# turn groups of 4, then 3, thne 2 spaces into tabs. 
a = re.sub(r'    ', r'\t', input_str)
b = re.sub(r'   ', r'\t', a)
c = re.sub(r'  ', r'\t', b)
escaped_input_str = escape_json_string(c)


sys.stdout.write(escaped_input_str + "\n")