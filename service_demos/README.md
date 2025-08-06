# Service Demonstrations & Automation Layer

This directory contains examples and fully-functional services that demonstrate how to use the **Dynamic Agent Workflow Engine** as the core of long-running, autonomous, and event-driven applications.

While the core engine can be used for single, command-line tasks, its true power is realized when it's integrated into a service-oriented architecture. The examples here provide the blueprints for doing just that.

## `server_demo.py`

A simple demonstration of how to wrap the core `exec_agent` function in a basic web server (e.g., using Flask or FastAPI). This shows how you could expose your entire library of agents as a private or public API.

**To Run:**
```bash
# (Install requirements first, e.g., pip install flask)
python server_demo.py
```

## The `queue/` Directory: An Advanced Job Dispatcher Service

This is a complete, production-ready blueprint for an asynchronous job processing system. It turns the workflow platform into a true automation fabric, capable of handling a queue of jobs submitted from various sources.

### Architecture

The system follows a robust producer-consumer model:

1.  **Producers (How jobs get submitted):**
    *   `file_watcher.py`: A service that monitors directories. When a new file (e.g., a PDF or CSV) is dropped into a watched folder, it creates a "job ticket."
    *   `scheduler.py`: A CRON-like service that submits jobs at scheduled times based on a `sched.json` definition.
    *   (Can be easily extended with an API endpoint or a message queue listener).

2.  **The Queue Server (`queue_server.py`):**
    *   This is the central hub. It runs a simple, in-memory queue that receives job tickets from all producers.
    *   It manages the state of all jobs (`PENDING`, `RUNNING`, `COMPLETED`, `FAILED`).

3.  **The Dispatcher & Workers (`dispatcher.py`):**
    *   This is the "engine room." It starts a pool of worker processes.
    *   Each worker constantly asks the queue server for a new job.
    *   When a worker receives a job, it calls the core **`exec_agent`** function from the main platform library to execute the requested workflow.
    *   Upon completion, it reports the final status back to the queue server.

### How to Run the Demo Service

The service is designed to be run as a set of coordinated processes, managed by shell scripts.

1.  **Configure:**
    *   Review `watch_config.json` to see which directories the `file_watcher` will monitor and which workflows will be triggered.
    *   Review `sched.json` to see which jobs are scheduled to run automatically.

2.  **Start the Services:**
    *   Use the provided `start.sh` script. This will launch the `queue_server.py`, `dispatcher.py`, `file_watcher.py`, and `scheduler.py` in the background.
    ```bash
    ./start.sh
    ```

3.  **Submit Jobs:**
    *   **Manually:** Drop a file into one of the `watched_...` directories.
    *   **Automatically:** Wait for the scheduler to trigger a job.

4.  **Monitor Status:**
    *   The queue server runs a small web interface to view the status of jobs. Open `templates/status.html` in your browser or hit the server's status endpoint.

5.  **Stop the Services:**
    ```bash
    ./stop.sh
    ```

This advanced demonstration proves that the core workflow engine is not just a tool, but a powerful, high-performance library that can serve as the reliable foundation for complex, scalable, and fully automated enterprise-grade services.
