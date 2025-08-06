
# Dynamic Agent Workflow: A Generative Development Platform

This project is a complete, high-performance ecosystem for creating, managing, and executing automated workflows. It is built on a simple yet profound architectural principle: **complexity should be emergent, not designed.**

By starting with a tiny, stable, and universal core engine, the platform allows for the organic growth of sophisticated capabilities, including a full graphical IDE, an integrated testing framework, and an autonomous job dispatching service. The entire system is driven by a `config.json` file that centralizes not just the logic, but the documentation, tests, and even the UI behavior of every component.

This is not just a workflow tool; it is a case study in **Emergent Development**, where a powerful architecture becomes a generative force, allowing a single developer to build an entire application ecosystem in days, not months.

## Core Philosophy: The Fractal Agent

The entire system is built on a single, powerful axiom: **An agent IS a workflow, and a workflow IS an agent.**

There are only three fundamental primitives:
1.  **`proc`**: An atomic unit of action (any Python function).
2.  **`template`**: An atomic unit of data transformation (string formatting).
3.  **`workflow`**: A composite unit that orchestrates other agents.

Crucially, all three share the exact same interface (`inputs` -> `outputs`). This **fractal self-similarity** is the key to the system's power. It means any sequence of steps can be seamlessly encapsulated and refactored into a new, reusable workflow agent, allowing for infinite, nested composition.

## The Ecosystem: How The Pieces Fit Together

This repository is not just one program; it's a collection of integrated services that all operate on the central `config.json` file.

```
.
├── config.json                 # <<< The single source of truth for the entire system
│
├── dynamic_workflows_agents.py # <<< The Core Engine (Command-Line Executor)
│
├── editor/                     # <<< The IDE (Visual Development Environment)
│   ├── editor_app.py           # The main GUI application to visually build and edit agents.
│   └── ...
│
├── unit_testing/               # <<< The QA Service
│   └── test_runner.py          # A command-line tool to run all agent tests and generate a report.
│
├── service_demos/              # <<< The Automation & Service Layer
│   ├── queue/                  # An advanced, autonomous job dispatcher service.
│   └── ...
│
└── ... (docs, examples, utilities)
```

### 1. The Core Engine (`dynamic_workflows_agents.py`)

This is the heart of the platform. It's a lightweight, high-performance Python script that can execute any agent or workflow defined in `config.json` directly from the command line.

*   **Function:** Parses `config.json`, builds a command-line interface on the fly for any agent, and executes the requested workflow.
*   **Key Feature:** The `exec_agent` function provides a unified entry point that transparently promotes simple `proc` and `template` agents into executable workflows, enforcing the core "fractal agent" philosophy.

### 2. The IDE (`editor/`)

This is the visual front-end for the entire ecosystem. It's a full-featured Integrated Development Environment (IDE) for creating, editing, testing, and managing your agents.

*   **Function:** Provides a rich, graphical interface to manipulate `config.json`.
*   **Key Features:**
    *   **Visual Workflow Builder:** Drag-and-drop steps to build workflows.
    *   **"Promote to Workflow":** The killer feature. Select any block of steps and automatically refactor them into a new, reusable workflow agent.
    *   **Integrated "Run & Test" Modal:** Execute any agent directly from the IDE and create, save, and run unit tests on the fly.

### 3. The QA Service (`unit_testing/test_runner.py`)

This is the automated quality assurance framework for the platform. It provides a way to validate the correctness and reliability of every agent in your library at once.

*   **Function:** Scans `config.json` for all agents that have `run_config` test definitions, executes them, and validates the results.
*   **Key Feature:** Generates a professional, self-contained HTML report that details the pass/fail status of every test, providing deep diagnostic information for any failures.

### 4. The Automation Layer (`service_demos/`)

This demonstrates how to use the core engine as the heart of long-running, autonomous services. The `queue/` directory contains a sophisticated job dispatching system.

*   **Function:** Watches directories for new job files, adds them to a queue, and uses a pool of workers to execute the requested workflows via the core engine.
*   **Key Feature:** Shows how the platform can be used for asynchronous, event-driven automation, transforming it from a simple tool into a true automation fabric.

## Getting Started

To get started with the platform, all essential components must be in the same root directory.

1.  **Core Files:** The central `config.json` file and the `dynamic_workflows_agents.py` engine are the bare minimum required for command-line execution.
2.  **Running the IDE:** To use the visual editor, navigate into the `editor/` directory and run the `editor_app.py` script. It will automatically load and modify the `config.json` file from the parent directory.
    ```bash
    cd editor
    python editor_app.py
    ```
3.  **Running Tests:** To run the entire test suite for your configured agents, use the `test_runner.py` script from the `unit_testing/` directory.
    ```bash
    cd unit_testing
    python test_runner.py --config ../config.json
    ```

This project is a living demonstration of how a commitment to architectural simplicity can lead to the emergent creation of a powerful and sophisticated software ecosystem. Welcome.
