# file_watcher.py
import requests
import json
import time
import os
import glob
from datetime import datetime

QUEUE_SERVER_URL = "http://127.0.0.1:5000"
CONFIG_FILE = "watch_config.json"
STATE_FILE = ".watcher_state.json"

def load_config():
    """Loads the watcher configuration from the JSON file."""
    print("[*] Loading watcher configuration from", CONFIG_FILE)
    try:
        with open(CONFIG_FILE, 'r') as f:
            all_watchers = json.load(f)
            enabled_watchers = [w for w in all_watchers if w.get("enabled", False)]
            print(f"  - Found {len(enabled_watchers)} enabled watchers.")
            return enabled_watchers
    except Exception as e:
        print(f"[X] Error loading config file: {e}")
        return []

def load_state():
    """Loads the last run times for each watcher from the state file."""
    if not os.path.exists(STATE_FILE):
        return {}
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[X] Error loading state file: {e}")
        return {}

def save_state(state):
    """Saves the current run times to the state file."""
    try:
        with open(STATE_FILE, 'w') as f:
            json.dump(state, f, indent=2)
    except Exception as e:
        print(f"[X] Error saving state file: {e}")

def schedule_job(workflow, initial_tape, retry_count):
    """Sends a job request to the queue server."""
    payload = {
        "workflow": workflow,
        "initial_tape": initial_tape,
        "retry_count": retry_count
    }
    try:
        response = requests.post(f"{QUEUE_SERVER_URL}/schedule", json=payload)
        response.raise_for_status()
        print(f"  - Successfully scheduled job for workflow '{workflow}' with tape: {initial_tape}")
    except requests.exceptions.RequestException as e:
        print(f"[X] Failed to schedule job for workflow '{workflow}': {e}")

def check_directory(watcher_config, last_check_time):
    """
    Checks a directory for new files based on the last check time.
    """
    path = watcher_config['path']
    pattern = watcher_config['pattern']
    workflow = watcher_config['workflow']
    retry_count = watcher_config.get('retry_count', 3)

    if not os.path.isdir(path):
        print(f"[!] Watched directory not found: {path}. Skipping.")
        return

    # Construct the full search pattern
    search_pattern = os.path.join(path, pattern)
    
    # Find all matching files
    try:
        matching_files = glob.glob(search_pattern)
    except Exception as e:
        print(f"[X] Error searching for files with pattern '{search_pattern}': {e}")
        return

    new_files_found = 0
    for file_path in matching_files:
        try:
            # Get the file's modification time (as a Unix timestamp)
            mod_time = os.path.getmtime(file_path)

            # Check if the file was modified AFTER the last check
            if mod_time > last_check_time:
                new_files_found += 1
                
                # The initial tape always includes the path to the discovered file
                initial_tape = {
                    "triggered_by": "file_watcher",
                    "watcher_id": watcher_config['id'],
                    "file_path": os.path.abspath(file_path)
                }
                
                schedule_job(workflow, initial_tape, retry_count)
        
        except FileNotFoundError:
            # The file might have been deleted between glob and getmtime, which is fine.
            continue
        except Exception as e:
            print(f"[X] Error processing file '{file_path}': {e}")

    if new_files_found > 0:
        print(f"[*] Watcher '{watcher_config['id']}': Found and scheduled {new_files_found} new file(s).")

def main():
    print("[+] File Watcher Service started.")
    
    watchers = load_config()
    state = load_state()

    # Initialize last check time and next check time for each watcher
    for w in watchers:
        watcher_id = w['id']
        # If we have a saved state, use it. Otherwise, use the current time.
        # This ensures we only process files created/modified AFTER the watcher starts for the first time.
        if watcher_id not in state:
            state[watcher_id] = {'last_check': time.time()}
        
        state[watcher_id]['next_check'] = time.time() # Check immediately on start

    if not watchers:
        print("[!] No enabled watchers found in config. Exiting.")
        return
        
    try:
        while True:
            now = time.time()
            for watcher in watchers:
                watcher_id = watcher['id']
                
                if now >= state[watcher_id]['next_check']:
                    print(f"--- Running check for '{watcher_id}' ---")
                    
                    last_check_timestamp = state[watcher_id]['last_check']
                    check_directory(watcher, last_check_timestamp)
                    
                    # Update state for the next run
                    state[watcher_id]['last_check'] = now
                    state[watcher_id]['next_check'] = now + watcher['check_interval_seconds']
                    
                    save_state(state) # Persist state after every successful check

            time.sleep(1) # Main loop sleep to prevent high CPU usage

    except KeyboardInterrupt:
        print("\n[*] File Watcher Service shutting down.")
    finally:
        save_state(state) # Always save state on exit

if __name__ == "__main__":
    main()