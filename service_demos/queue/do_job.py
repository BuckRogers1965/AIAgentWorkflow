# do_job.py
import requests
import argparse
import json
import sys

QUEUE_SERVER_URL = "http://127.0.0.1:5000"

def main():
    parser = argparse.ArgumentParser(description="Manually schedule a workflow job.")
    parser.add_argument("workflow", help="The name of the workflow file to run.")
    parser.add_argument("tape", help="The initial tape as a JSON string. E.g., '{\"key\":\"value\"}'")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries on failure.")
    
    args = parser.parse_args()

    try:
        initial_tape = json.loads(args.tape)
    except json.JSONDecodeError:
        print("Error: Invalid JSON string for 'tape'.", file=sys.stderr)
        sys.exit(1)
        
    payload = {
        "workflow": args.workflow,
        "initial_tape": initial_tape,
        "retry_count": args.retries
    }

    try:
        response = requests.post(f"{QUEUE_SERVER_URL}/schedule", json=payload)
        response.raise_for_status() # Raise an exception for bad status codes
        print("Job scheduled successfully!")
        print(response.json())
    except requests.exceptions.RequestException as e:
        print(f"Error scheduling job: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()