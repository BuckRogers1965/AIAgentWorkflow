# queue_server.py

# --- DIAGNOSTIC CHECK ---
# This code runs first to identify the source of the import error.
import sys
import importlib.util

try:
    # Try to find the spec for the standard library 'queue'
    spec = importlib.util.find_spec('queue')
    if spec:
        print("--- DIAGNOSTIC INFO ---", flush=True)
        print(f"Found 'queue' module at: {spec.origin}", flush=True)
        if sys.path[0] in spec.origin:
             print("!!! CRITICAL WARNING: The 'queue' module is being loaded from the current project directory.", flush=True)
             print("!!! Please ensure no file is named 'queue.py' in this folder.", flush=True)
        else:
             print("OK: 'queue' module appears to be the correct standard library version.", flush=True)
        print("--- END DIAGNOSTIC ---", flush=True)
    else:
        print("!!! CRITICAL ERROR: Could not find the standard 'queue' library at all.", flush=True)
except Exception as e:
    print(f"!!! An unexpected error occurred during diagnostic: {e}", flush=True)
# --- END DIAGNOSTIC CHECK ---


from flask import Flask, request, jsonify
from collections import deque
import uuid
import threading
import time
from datetime import datetime
import html

app = Flask(__name__)

# --- In-memory data stores (for simplicity) ---
PENDING_JOBS = deque()
RUNNING_JOBS = {}
COMPLETED_JOBS = deque(maxlen=100)
FAILED_JOBS = deque(maxlen=100)

data_lock = threading.Lock()

@app.route('/schedule', methods=['POST'])
def schedule_job():
    job_data = request.json
    if not job_data or 'workflow' not in job_data:
        return jsonify({"error": "Invalid job data"}), 400
    job_id = str(uuid.uuid4())
    job = {
        "id": job_id, "workflow": job_data['workflow'],
        "initial_tape": job_data.get('initial_tape', {}), "status": "pending",
        "retry_count": job_data.get('retry_count', 3), "submitted_at": datetime.utcnow().isoformat()
    }
    with data_lock:
        PENDING_JOBS.append(job)
    print(f"[*] Job {job_id} scheduled for workflow '{job['workflow']}'.")
    return jsonify({"message": "Job scheduled successfully", "job_id": job_id}), 201

@app.route('/get_job', methods=['GET'])
def get_job():
    with data_lock:
        if not PENDING_JOBS:
            return jsonify({}), 204
        job = PENDING_JOBS.popleft()
        job['status'] = 'running'
        job['started_at'] = datetime.utcnow().isoformat()
        RUNNING_JOBS[job['id']] = job
        print(f"[>] Job {job['id']} dispatched.")
        return jsonify(job)

@app.route('/update_status/<job_id>', methods=['POST'])
def update_status(job_id):
    update_data = request.json
    status = update_data.get('status')
    with data_lock:
        if job_id not in RUNNING_JOBS:
            return jsonify({"error": "Job not found or not running"}), 404
        job = RUNNING_JOBS.pop(job_id)
        job['completed_at'] = datetime.utcnow().isoformat()
        job['final_status'] = status
        if status == 'success':
            COMPLETED_JOBS.appendleft(job)
            print(f"[+] Job {job_id} completed successfully.")
        elif status == 'failed':
            if job['retry_count'] > 0:
                job['retry_count'] -= 1
                job['status'] = 'pending'
                PENDING_JOBS.append(job)
                print(f"[!] Job {job_id} failed. Retrying ({job['retry_count']} left)...")
            else:
                FAILED_JOBS.appendleft(job)
                print(f"[X] Job {job_id} failed permanently.")
        else:
            return jsonify({"error": "Invalid status"}), 400
    return jsonify({"message": "Status updated"}), 200

@app.route('/status')
def status_page():
    with data_lock:
        pending = list(PENDING_JOBS)
        running = list(RUNNING_JOBS.values())
        completed = list(COMPLETED_JOBS)
        failed = list(FAILED_JOBS)
    def format_job(j):
        return f"<div class='job'>{html.escape(str(j))}</div>"
    html_template = """
    <html><head><title>Queue Status</title><meta http-equiv="refresh" content="5">
    <style>body{{font-family:monospace;padding:1em;}} h2{{border-bottom:1px solid #ccc;}} .job{{border:1px solid #eee;padding:5px;margin-bottom:5px;background-color:#f9f9f9;word-wrap:break-word;}}</style>
    </head><body><h1>Job Queue Status</h1><p><b>Last updated:</b> {now}</p>
    <h2>Running Jobs ({running_count})</h2>{running_jobs}
    <h2>Pending Jobs ({pending_count})</h2>{pending_jobs}
    <h2>Completed Jobs ({completed_count})</h2>{completed_jobs}
    <h2>Failed Jobs ({failed_count})</h2>{failed_jobs}
    </body></html>""".format(
        now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        running_count=len(running), running_jobs=''.join(map(format_job, running)),
        pending_count=len(pending), pending_jobs=''.join(map(format_job, pending)),
        completed_count=len(completed), completed_jobs=''.join(map(format_job, completed)),
        failed_count=len(failed), failed_jobs=''.join(map(format_job, failed))
    )
    return html_template

if __name__ == '__main__':
    print("[+] Queue Server starting on http://127.0.0.1:5000")
    print("[+] Status page available at http://127.0.0.1:5000/status")
    app.run(host='127.0.0.1', port=5000)