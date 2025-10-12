# service_demos/mcp_client_demo.py
import socket
import json
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(
        description="A simple client for the MCP Agent Workflow service.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="""
Example Usage:
  # List all available agents
  python mcp_client_demo.py --request '{"command": "LIST_AGENTS"}'

  # Execute a simple agent
  python mcp_client_demo.py --request '{"command": "EXECUTE_AGENT", "agent": "append_text", "params": {"whole_text": "Hello, ", "part_text": "World!"}}'
"""
    )
    parser.add_argument('--host', default='127.0.0.1', help='The server host.')
    parser.add_argument('--port', type=int, default=5001, help='The server port.')
    parser.add_argument('--request', required=True, help='The JSON request string to send to the server.')
    args = parser.parse_args()

    try:
        # Validate the JSON request string before sending
        request_dict = json.loads(args.request)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON provided in --request argument.\n{e}", file=sys.stderr)
        sys.exit(1)

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            print(f"Connecting to {args.host}:{args.port}...")
            s.connect((args.host, args.port))
            
            print("Sending request...")
            # Send the JSON request, encoded and with a newline
            s.sendall((args.request + '\n').encode('utf-8'))
            
            print("Waiting for response...")
            # Receive the response
            response_data = b""
            while True:
                chunk = s.recv(4096)
                if not chunk:
                    break
                response_data += chunk
                if b'\n' in chunk:
                    break
            
            response_str = response_data.decode('utf-8').strip()
            
            print("\n--- SERVER RESPONSE ---")
            try:
                # Pretty-print the JSON response
                response_json = json.loads(response_str)
                print(json.dumps(response_json, indent=2))
            except json.JSONDecodeError:
                print("Received non-JSON response:")
                print(response_str)

    except ConnectionRefusedError:
        print(f"Error: Connection refused. Is the MCP service running on {args.host}:{args.port}?", file=sys.stderr)
    except Exception as e:
        print(f"An error occurred: {e}", file=sys.stderr)

if __name__ == '__main__':
    main()