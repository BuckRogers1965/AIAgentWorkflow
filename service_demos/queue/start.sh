#!/bin/bash

# A script to start all components of the workflow orchestration system.

echo "[*] Ensuring necessary directories exist..."
mkdir -p watched_pdfs
mkdir -p watched_csvs

echo "[*] Cleaning up any previous log and pid files..."
rm -f *.log
rm -f *.pid

echo "[+] Starting Queue Server..."
nohup python3.11 queue_server.py > log/queue_server.log 2>&1 &
echo $! > queue_server.pid
sleep 2 # Give the server a moment to start up

echo "[+] Starting Scheduler..."
nohup python3.11 scheduler.py > log/scheduler.log 2>&1 &
echo $! > scheduler.pid

echo "[+] Starting File Watcher..."
nohup python3.11 file_watcher.py > log/file_watcher.log 2>&1 &
echo $! > file_watcher.pid

echo "[+] Starting 3 Dispatcher Workers..."
nohup python3.11 dispatcher.py > log/dispatcher_1.log 2>&1 &
echo $! > dispatcher_1.pid
nohup python3.11 dispatcher.py > log/dispatcher_2.log 2>&1 &
echo $! > dispatcher_2.pid
nohup python3.11 dispatcher.py > log/dispatcher_3.log 2>&1 &
echo $! > dispatcher_3.pid

echo "[*] All services started in the background."
echo "    - Logs are being written to log/*.log files."
    echo "    - Process IDs are stored in *.pid files."
echo "    - Monitor the system at http://127.0.0.1:5000/status"
echo ""

# Wait a moment for the system to be fully ready
sleep 1

# --- Manually queue a job to test the system ---
echo "[*] Queuing a manual test job..."
python3.11 do_job.py "Manual_Onboarding_Workflow" '{"user_id":"test-123", "department":"sales"}' --retries 2

echo ""
echo "[*] Test job has been queued. Watch the logs and the status page."
echo "[*] To stop all services, run ./stop.sh"
