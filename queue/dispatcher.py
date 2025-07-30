# dispatcher.py
import requests
import time
import random
import os

QUEUE_SERVER_URL = "http://127.0.0.1:5000"

def execute_workflow(workflow_name, initial_tape):
    """
    This is a MOCK execution engine.
    In your real system, this function would load the workflow JSON,
    and run it step-by-step, managing the 'tape'.
    """
    worker_id = os.getpid()
    print(f"  [Worker {worker_id}] Executing workflow '{workflow_name}'...")
    print(f"  [Worker {worker_id}] Initial Tape: {initial_tape}")
    
    # Simulate doing work
    time.sleep(random.randint(2, 6))
    
    # Simulate a chance of failure to test retries
    if random.random() < 0.3: # 30% chance of failure
        print(f"  [Worker {worker_id}] Workflow '{workflow_name}' FAILED.")
        return "failed"
    else:
        print(f"  [Worker {worker_id}] Workflow '{workflow_name}' SUCCEEDED.")
        return "success"

def main():
    worker_id = os.getpid()
    print(f"[+] Dispatcher worker {worker_id} started. Polling for jobs...")
    while True:
        try:
            # 1. Ask for a job
            response = requests.get(f"{QUEUE_SERVER_URL}/get_job", timeout=10)
            
            if response.status_code == 200: # We got a job
                job = response.json()
                print(f"[>] Worker {worker_id} picked up job {job['id']}.")

                # 2. Execute the workflow
                status = execute_workflow(job['workflow'], job['initial_tape'])

                # 3. Report back the status
                requests.post(
                    f"{QUEUE_SERVER_URL}/update_status/{job['id']}",
                    json={"status": status}
                )
            elif response.status_code == 204: # No jobs available
                time.sleep(3) # Wait before polling again
            else:
                print(f"[!] Error contacting queue server: {response.status_code}")
                time.sleep(10)

        except requests.exceptions.RequestException as e:
            print(f"[X] Cannot connect to queue server: {e}")
            time.sleep(15)

if __name__ == "__main__":
    main()