# Dynamic Agent Workflow IDE

**TL&DR**

    ```bash
    # From within the editor/ directory
    python editor_app.py --config ../config.json --lib-path ..
    ```
    *   `--config`: Specifies the path to your main `config.json` file.
    *   `--lib-path`: Specifies the directory containing `dynamic_workflows_agents.py`.



Welcome to the visual development environment for the Dynamic Agent Workflow platform. This application is a powerful, graphical tool for creating, editing, managing, and testing agents and workflows defined in your `config.json` file.

This IDE is a testament to the "Emergent Development" philosophy. It is not just a front-end for the core engine; it is a sophisticated application that arises from the simple, self-describing nature of the agent architecture. Features like the "Promote to Workflow" refactoring tool and the integrated testing framework are natural consequences of the platform's core fractal design.

## Key Features

*   **Visual Workflow Builder:** Drag-and-drop agents to build and reorder complex workflows.
*   **Intelligent Agent Editors:** Context-aware forms for `proc`, `template`, and `workflow` agents.
*   **Live Run & Test Environment:** Execute any agent with custom inputs and build a persistent suite of unit tests directly within the UI.
*   **One-Click Refactoring:** Select any block of steps in a workflow and automatically encapsulate them into a new, reusable workflow agent.
*   **Fully Themeable Interface:** Customize every color, font, and style element of the IDE to your liking using a simple `themes.json` file.

## Architecture: A Modular, Component-Based UI

To enhance maintainability and clarity, the IDE is broken down into a set of focused, single-responsibility Python modules. This modular structure separates the application's core logic from its various UI components.

### Directory Structure

```
editor/
├── editor_app.py           # Main application entry point (setup and launch).
│
├── config_manager.py       # Manages all interactions with the config.json file.
├── ui_app_shell.py         # The main App class and top-level window layout.
├── ui_run_modal.py         # The "Run Agent" modal and its integrated testing UI.
│
├── ui_editors.py           # Contains all editor forms (Proc, Template, JSON) and their components.
├── ui_step_editor.py       # The modal for editing a single workflow step's parameters.
├── ui_theme_editor.py      # The modal for creating and managing visual themes.
├── workflow_editor.py      # The specific editor frame for workflow agents.
│
├── theme_manager.py        # The back-end logic for loading and saving themes.
├── themes.json             # (Optional) User-defined themes are stored here.
│
└── requirements.txt        # Python dependencies for the editor.
```

### Core Components

1.  **`editor_app.py` (The Launcher):**
    This is the executable entry point. Its only job is to handle command-line arguments (`--config`, `--lib-path`), load the core workflow engine, and launch the main application shell.

2.  **`ui_app_shell.py` (The Shell):**
    This contains the main `App` class. It builds the main window, lays out the primary panels (agent list, editor panel), and acts as the central orchestrator for all other UI components. It also manages the `ThemeManager` instance.

3.  **The Editor Modules (`ui_editors.py`, `workflow_editor.py`):**
    These files contain the specialized `CTkFrame` classes that provide the user interface for editing each type of agent. They are designed to be "dumb" renderers that receive agent data and display it in a structured way.

4.  **The Modal Modules (`ui_run_modal.py`, `ui_step_editor.py`, `ui_theme_editor.py`):**
    Each of these files defines a self-contained `CTkToplevel` window that pops up to perform a specific, focused task, such as running an agent or editing a theme.

## The Theming Engine

The IDE's appearance is not hard-coded. It is driven by a powerful and flexible theming engine that allows for deep customization.

*   **`theme_manager.py`:** This class handles all logic for loading, creating, and saving visual themes.
*   **`themes.json`:** Themes are stored in this JSON file. Users can create their own themes here. If the file is missing, the application will create it with default "Dark" and "Light" themes.
*   **Granular Control:** The theme file allows you to specify colors, font families, and font sizes for the general UI as well as for specific contexts like the **Code Editor**, **Template Editor**, and **Step Parameter** fields.
*   **Live Previews:** The **Theme Editor** (accessible via the "T" button) allows you to edit themes and see your changes applied to the entire application in real-time.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
2.  **Run the Application:**
    Execute the main entry point. The IDE is designed to be run from its own directory, but can be configured to find your core project files elsewhere.
    ```bash
    # From within the editor/ directory
    python editor_app.py --config ../config.json --lib-path ..
    ```
    *   `--config`: Specifies the path to your main `config.json` file.
    *   `--lib-path`: Specifies the directory containing `dynamic_workflows_agents.py`.

This modular and themeable architecture ensures that the IDE is not only powerful but also maintainable, extensible, and a pleasure to use.
