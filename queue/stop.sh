#!/bin/bash

# A script to stop all running components of the workflow system.

echo "[*] Stopping all services..."

# Check for pid files and kill the processes
for pidfile in *.pid; do
    if [ -f "$pidfile" ]; then
        pid=$(cat "$pidfile")
        if ps -p "$pid" > /dev/null; then
            echo "    - Stopping process with PID $pid from $pidfile..."
            kill "$pid"
        else
            echo "    - Process with PID $pid from $pidfile not found (already stopped)."
        fi
        rm "$pidfile"
    fi
done

echo "[*] Cleanup complete."