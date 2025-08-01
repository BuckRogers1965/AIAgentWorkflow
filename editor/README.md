# AI Agent Workflow Editor

A powerful, feature-complete graphical user interface (GUI) for creating, managing, and editing a declarative, agent-based AI workflow system. This application provides a robust and user-friendly way to edit the central `config.json` file that defines the behavior of the entire AI workflow engine.

Built with Python and CustomTkinter, this editor replaces manual JSON editing with a structured, intuitive interface, preventing syntax errors and dramatically speeding up development of complex AI workflows.

## Features

-   **Full Agent Management:**
    -   **Create:** Easily create new agents of type `workflow`, `proc`, or `template`.
    -   **Rename:** Edit agent names directly. The application automatically finds and updates all references to the old name across all workflow steps.
    -   **Delete:** Safely delete agents with a dedicated button. Includes a crucial **dependency check** to prevent deleting an agent that is currently used by a workflow.
-   **Dynamic Editor Panels:**
    -   The main editor view intelligently builds a unique UI based on the selected agent's type.
    -   **Workflow Editor:** A dedicated view for managing workflow steps, including reordering, adding, and deleting steps.
    -   **Proc Editor:** A form that includes a full-featured **code editor** with Python syntax highlighting and line numbers for the `function_def`.
    -   **Template Editor:** A form with a large textbox for writing and editing prompts.
    -   **JSON Editor:** A fallback raw JSON editor for any unrecognized agent types, ensuring forward compatibility.
-   **Robust Workflow Step Editor:**
    -   **True Modal Dialog:** When editing a step, the main window is locked and grayed out, forcing focus on the current task.
    -   **Two-Pane Layout:** A "Toolbox" on the left shows a clickable list of available optional inputs for the agent, and the main form for editing parameters is on the right.
    -   **Full Key-Value Editing:** Correctly edit output variable names, required parameter values, and both the keys and values of optional parameters.
    -   **Transactional Edits:** All changes are made on a temporary copy. They are only committed if you click **[ Save ]**, and are completely discarded if you click **[ Cancel ]**.
-   **Clean & Modern UI:**
    -   Built with CustomTkinter for a modern look and feel that respects system dark/light modes.
    -   Clear visual cues, including agent type icons (`[W]`, `[P]`, `[T]`, `[J]`) and intuitive button layouts.
    -   Non-blocking "toast" notifications for a smooth user experience.

## Prerequisites

-   Python 3.9+
-   `pip` (Python's package installer)

#### macOS Specific Prerequisite
If you are on macOS and installed Python via Homebrew, you may be missing the necessary Tkinter support. To fix this, run:
```bash
brew install python-tk
```
Then, verify your installation by running `python3 -m tkinter`. If a small window appears, you are ready to proceed.

## Installation & Setup

1.  **Clone the repository or download the files** into a new directory.
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```

2.  **Create and activate a virtual environment** (recommended):
    *   On macOS / Linux:
        ```bash
        python3 -m venv venv
        source venv/bin/activate
        ```
    *   On Windows:
        ```bash
        python -m venv venv
        .\venv\Scripts\activate
        ```

3.  **Install the required dependencies** using the `requirements.txt` file:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

With your virtual environment activated, run the main application file from your terminal:

```bash
python editor_app.py
```

The editor will launch and automatically create a `config.json` file if one does not exist.

## File Structure

-   `editor_app.py`: The main application file containing all the GUI code and logic.
-   `config_manager.py`: The backend "brain" of the application. It handles all loading, saving, and manipulation of the `config.json` file and is completely independent of the UI.
-   `config.json`: The central configuration file for your AI workflow system. This is the file that the editor reads from and writes to.
-   `requirements.txt`: A list of the Python packages required to run the editor.