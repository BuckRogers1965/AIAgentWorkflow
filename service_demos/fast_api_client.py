
import argparse
import json
import sys
import requests

def list_agents(args):
    """
    Fetches and lists all available agents from the service's /agents endpoint.
    """
    url = f"http://{args.host}:{args.port}/agents"
    print(f"Fetching available agents from {url}...")

    try:
        response = requests.get(url)
        
        if response.status_code != 200:
            print(f"Error: Received status code {response.status_code}", file=sys.stderr)
            print(f"Response: {response.text}", file=sys.stderr)
            sys.exit(1)

        data = response.json()
        agents = data.get('available_agents', [])
        
        print(f"\n--- Available Agents (Contract: {data.get('service_contract', 'N/A')}) ---")
        if not agents:
            print("No agents available under this service contract.")
            return

        for agent in agents:
            print(f"\n- Name: {agent.get('name', 'N/A')}")
            print(f"  Help: {agent.get('help', 'N/A')}")
            print(f"  Inputs: {agent.get('inputs', [])}")
            if agent.get('optional_inputs'):
                print(f"  Optional Inputs: {agent.get('optional_inputs', [])}")
            print(f"  Outputs: {agent.get('outputs', [])}")

    except requests.exceptions.ConnectionError:
        print(f"Error: Connection refused. Is the FastAPI service running on {args.host}:{args.port}?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)

def execute_agent(args):
    """
    Sends a request to the FastAPI service to execute a specific agent.
    """
    url = f"http://{args.host}:{args.port}/execute{args.agent_name}"
    
    try:
        params_dict = json.loads(args.params)
        if not isinstance(params_dict, dict):
            raise ValueError("Params must be a JSON object.")
    except (json.JSONDecodeError, ValueError) as e:
        print(f"Error: Invalid JSON in --params argument. Please provide a valid JSON object string.", file=sys.stderr)
        print(f"Details: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Executing agent '{args.agent_name}' on http://{args.host}:{args.port}...")
    
    try:
        response = requests.post(url, json=params_dict)
        
        print(f"\n--- SERVER RESPONSE (Status Code: {response.status_code}) ---")
        
        # Pretty-print the JSON response
        try:
            response_json = response.json()
            print(json.dumps(response_json, indent=2))
        except json.JSONDecodeError:
            print("Received non-JSON response:")
            print(response.text)

    except requests.exceptions.ConnectionError:
        print(f"Error: Connection refused. Is the FastAPI service running on {args.host}:{args.port}?", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="A command-line client for the FastAPI Agent Workflow service.",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument('--host', default='127.0.0.1', help='The server host (default: 127.0.0.1).')
    parser.add_argument('--port', type=int, default=8000, help='The server port (default: 8000).')
    
    subparsers = parser.add_subparsers(dest='command', required=True, help='Available commands')

    # --- List Command ---
    parser_list = subparsers.add_parser('list', help='Discover available agents via the interactive API docs.')
    parser_list.set_defaults(func=list_agents)

    # --- Execute Command ---
    parser_execute = subparsers.add_parser(
        'execute', 
        help='Execute a specific agent.',
        epilog="""
Example Usage:
  # Execute the 'append_text' agent with required parameters
  python fast_api_client.py execute append_text --params '{"whole_text": "Hello, ", "part_text": "FastAPI!"}'
"""
    )
    parser_execute.add_argument('agent_name', help='The name of the agent to execute.')
    parser_execute.add_argument('--params', required=True, help='A JSON string containing the agent parameters (e.g., \'{\"key\": \"value\"}\').')
    parser_execute.set_defaults(func=execute_agent)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
