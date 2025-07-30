# Simple Workflow Orchestration System

This project is a lightweight, powerful, and scalable workflow automation system built in Python. It is designed around a decoupled, queue-centric architecture that allows for the scheduling, execution, and monitoring of complex, multi-step workflows.

The system's philosophy is built on **simplicity and emergent power**. A few simple, robust components work together to enable complex, resilient, and intelligent automation. It can trigger workflows based on time schedules (cron jobs) or file system events (directory watching).

## Core Architecture

The system is composed of several distinct, standalone components that communicate via a central queue server. This decoupled design ensures scalability and resilience.

1.  **Queue Server (`queue_server.py`)**: The central nervous system. A Flask-based web server that manages job queues (pending, running, completed, failed) via a simple REST API. It also hosts a real-time web dashboard for monitoring the entire system's state.

2.  **Dispatcher (`dispatcher.py`)**: The "worker." This script polls the queue server for jobs, executes the specified workflow (currently simulated), and reports the success or failure back to the server. You can run multiple dispatcher instances, even on different machines, to scale processing power horizontally.

3.  **Scheduler (`scheduler.py`)**: The time-based "foreman." This script watches a configuration file (`sched.json`) for cron-based job schedules. When a job is due, it adds a new job to the queue. It automatically hot-reloads its configuration when `sched.json` is modified.

4.  **File Watcher (`file_watcher.py`)**: The event-driven "sentry." This script monitors one or more directories for new files, as defined in `watch_config.json`. When a new file matching a specified pattern is detected, it adds a new job to the queue with the file's path as input.

5.  **Command-Line Tool (`do_job.py`)**: A manual utility for injecting a single, on-demand job into the queue. This is perfect for testing, manual interventions, or triggering workflows from other scripts.

### System Diagram



*(The Schedulers and Watchers are "Producers," adding jobs to the Queue. The Dispatchers are "Consumers," taking jobs from the Queue.)*

## Features

-   **Decoupled & Scalable**: Schedulers, watchers, and dispatchers are fully independent, enabling high throughput and horizontal scaling.
-   **Event-Driven Automation**: The file watcher can trigger complex workflows the moment a new file arrives.
-   **Time-based Scheduling**: Uses standard `cron` syntax for powerful, flexible scheduling.
-   **Manual Job Submission**: A command-line tool allows for easy on-demand job execution.
-   **Automatic Retries**: Jobs can be configured to automatically retry on failure, increasing system resilience.
-   **Real-time Monitoring**: A simple web dashboard provides a live view of all job queues and system performance.
-   **Hot-Reloading Configuration**: Modify the `sched.json` file, and the scheduler updates itself automatically without a restart.
-   **Stateful File Watching**: The file watcher remembers which files it has already processed, even after a restart, to prevent duplication.

## Setup and Installation

### Prerequisites

-   Python 3.8+

### Installation

1.  Clone this repository or download the files into a new project directory.
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  It is highly recommended to use a virtual environment:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```
3.  Install the required Python packages using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```
4.  Create the directories that the file watcher will monitor, as defined in `watch_config.json`:
    ```bash
    mkdir watched_pdfs
    mkdir watched_csvs
    ```

## How to Run

The system is designed to be run in multiple, separate terminal windows. The order is important: start the server first, then the workers, then the job creators.

**1. Start the Queue Server (Terminal 1)**

This is the central component and must be running first.

```bash
python queue_server.py
```
-   The API will be available at `http://127.0.0.1:5000`.
-   Open the **status dashboard** in your web browser: **`http://127.0.0.1:5000/status`**

**2. Start One or More Dispatchers (Terminal 2, 3, etc.)**

Open a new terminal for each dispatcher you want to run. These are the workers.

```bash
python dispatcher.py
```
-   The dispatcher will immediately start polling the queue for jobs to execute. You can run as many of these as you need.

**3. Start the Job Producers (Scheduler and/or Watcher)**

You can run either or both of these, depending on your needs.

**To run the time-based scheduler (Terminal 4):**
```bash
python scheduler.py
```
-   This will watch `sched.json` and queue jobs based on their cron schedule.

**To run the event-driven file watcher (Terminal 5):**
```bash
python file_watcher.py
```
-   This will watch the directories defined in `watch_config.json`. To test it, drop a new file (e.g., a PDF) into the corresponding directory.

**4. (Optional) Run a Job Manually (Another Terminal)**

Use the command-line tool to queue a job immediately.

```bash
# Usage: python do_job.py <workflow_name> '<json_tape>' [--retries N]

# Example:
python do_job.py "Manual_ETL_Process" "{\"source\":\"database_a\",\"destination\":\"data_lake\"}" --retries 1
```
-   Watch the terminals and the web dashboard to see your job get queued, picked up by a dispatcher, and processed in real-time.

## Customization

-   **Define Schedules**: Edit the `sched.json` file to create your own time-based jobs.
-   **Define Watchers**: Edit the `watch_config.json` file to watch different directories for different file patterns.
-   **Implement Real Workflows**: The `execute_workflow` function in `dispatcher.py` is currently a placeholder. Replace its `time.sleep()` logic with your actual workflow execution engine that reads and runs your workflow definition files.