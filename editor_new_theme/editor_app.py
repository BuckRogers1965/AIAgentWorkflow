# --- START OF FILE editor_app.py ---
import customtkinter as ctk
import argparse
import sys
import os
from tkinter import messagebox

# Import the main application window from its separate file.
from ui_app_shell import App

if __name__ == "__main__":
    # 1. Parse Arguments
    parser = argparse.ArgumentParser(description="A visual editor for the Dynamic Agent Workflow system.")
    parser.add_argument('--config', default='config.json', help='Path to the configuration file.')
    parser.add_argument('--lib-path', help='Path to the directory containing dynamic_workflows_agents.py.')
    args = parser.parse_args()

    # 2. Prepare Library Path
    if args.lib_path:
        sys.path.insert(0, os.path.abspath(args.lib_path))

    # 3. Attempt to Load the Core Engine
    core_lib = None
    try:
        import dynamic_workflows_agents
        from dynamic_workflows_agents import exec_agent, setup_depth_manager
        
        # On success, bundle the functions into a dictionary.
        core_lib = {
            "dynamic_workflows_agents": dynamic_workflows_agents,
            "exec_agent": exec_agent,
            "setup_depth_manager": setup_depth_manager
        }
    except ImportError:
        messagebox.showerror("Import Error",
                             "Could not import the core workflow engine from 'dynamic_workflows_agents.py'.\n\n"
                             "The 'Run' feature will be disabled.\n\n"
                             "Please ensure the file is in a standard location or specify its directory "
                             "using the --lib-path command-line option.")

    # 4. Initialize and Launch the Application
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    # Pass the config path AND the loaded library (or None) to the App.
    # This is the line that was causing the error because the App was not ready for it.
    app = App(config_path=args.config, core_lib=core_lib)
    
    app.mainloop()
