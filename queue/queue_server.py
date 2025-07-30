# queue_server.py
from flask import Flask, request, jsonify, render_template  # <-- IMPORT render_template
from collections import deque
import uuid
import threading
import time
from datetime import datetime

app = Flask(__name__)

# --- In-memory data stores ---
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
    """
    Renders the status page using a proper HTML template.
    This is the robust and correct way to generate HTML in Flask.
    """
    with data_lock:
        # Create copies of the data to pass to the template
        pending = list(PENDING_JOBS)
        running = list(RUNNING_JOBS.values())
        completed = list(COMPLETED_JOBS)
        failed = list(FAILED_JOBS)

    # Pass the data to the template for safe rendering
    return render_template(
        'status.html',
        now=datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        running_jobs=running,
        pending_jobs=pending,
        completed_jobs=completed,
        failed_jobs=failed
    )

if __name__ == '__main__':
    print("[+] Queue Server starting on http://127.0.0.1:5000")
    print("[+] Status page available at http://127.0.0.1:5000/status")
    app.run(host='127.0.0.1', port=5000)