# scheduler.py
import requests
import json
import time
from datetime import datetime
from croniter import croniter
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

QUEUE_SERVER_URL = "http://127.0.0.1:5000"
SCHEDULE_FILE = "sched.json"

# --- Global state for the scheduler ---
SCHEDULED_JOBS = []
LAST_RUN_TIMES = {}
RELOAD_CONFIG = True # Load config on first run

class ConfigChangeHandler(FileSystemEventHandler):
    """Sets a flag to reload the config when sched.json changes."""
    def on_modified(self, event):
        if event.src_path.endswith(SCHEDULE_FILE):
            global RELOAD_CONFIG
            RELOAD_CONFIG = True
            print("[*] Detected change in sched.json. Will reload on next cycle.")

def load_schedule():
    """Loads and validates the schedule from the JSON file."""
    global SCHEDULED_JOBS, RELOAD_CONFIG
    print("[+] Loading schedule from", SCHEDULE_FILE)
    try:
        with open(SCHEDULE_FILE, 'r') as f:
            all_jobs = json.load(f)
            # Filter for enabled jobs with a cron trigger
            SCHEDULED_JOBS = [
                job for job in all_jobs 
                if job.get("enabled", False) and job.get("trigger", {}).get("type") == "cron"
            ]
            print(f"  - Loaded {len(SCHEDULED_JOBS)} enabled cron jobs.")
    except Exception as e:
        print(f"[X] Error loading schedule file: {e}")
        SCHEDULED_JOBS = []
    RELOAD_CONFIG = False

def schedule_job(job_config):
    """Sends a job request to the queue server."""
    payload = {
        "workflow": job_config['workflow'],
        "initial_tape": job_config.get('initial_tape', {}),
        "retry_count": job_config.get('retry_count', 3)
    }
    try:
        requests.post(f"{QUEUE_SERVER_URL}/schedule", json=payload)
        print(f"  - Scheduled job '{job_config['id']}'")
    except requests.exceptions.RequestException as e:
        print(f"[X] Failed to schedule job '{job_config['id']}': {e}")


def main():
    # Start the watchdog to monitor config file changes
    event_handler = ConfigChangeHandler()
    observer = Observer()
    observer.schedule(event_handler, path='.', recursive=False)
    observer.start()
    print("[+] Scheduler started. Monitoring sched.json for changes.")

    base_time = datetime.now()
    
    while True:
        if RELOAD_CONFIG:
            load_schedule()

        now = datetime.now()
        # Check jobs every minute
        if now.minute != base_time.minute:
            base_time = now
            print(f"\n--- Checking schedule for {now.strftime('%Y-%m-%d %H:%M')} ---")
            for job in SCHEDULED_JOBS:
                cron_schedule = job['trigger']['value']
                job_id = job['id']
                
                # Check if this job should run now according to its cron schedule
                if croniter.match(cron_schedule, now):
                    # Check if it has already run this minute to prevent duplicates
                    if LAST_RUN_TIMES.get(job_id) != now.minute:
                        schedule_job(job)
                        LAST_RUN_TIMES[job_id] = now.minute

        time.sleep(1) # Main loop sleep
        
if __name__ == "__main__":
    main()