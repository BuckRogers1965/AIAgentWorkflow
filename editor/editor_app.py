# main_app.py
import customtkinter as ctk
from tkinter import messagebox, ttk
import json
import copy
import time
import re
import io
import logging
import base64
from functools import reduce 
import operator 
import sys
import os
import argparse
from config_manager import ConfigManager
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

from workflow_editor import WorkflowEditorFrame

# THIS BLOCK IS MOVED TO THE `if __name__ == "__main__":` section
# TO ALLOW COMMAND-LINE PATHS TO BE PROCESSED FIRST.
# Global placeholders are defined here.
dynamic_workflows_agents = None
exec_agent = None
create_temp_workflow = None
setup_depth_manager = None


# --- UTILITY FUNCTION ---
def get_nested(data, key_str):
    """Accesses a nested value in a dict using dot notation."""
    try:
        return reduce(operator.getitem, key_str.split('.'), data)
    except (KeyError, TypeError, AttributeError):
        return None
        
# --- Custom Code Editor Widget ---
class CTkCodeEditor(ctk.CTkFrame):
    def __init__(self, master, language="python", **kwargs):
        super().__init__(master, **kwargs)
        self.language_lexer = PythonLexer()
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        
        code_font = ctk.CTkFont(family="Courier", size=12)

        self.line_numbers = ctk.CTkTextbox(self, width=40, fg_color="#EAEAEA", text_color="#6A6A6A", font=code_font)
        self.line_numbers.grid(row=0, column=0, sticky="ns"); self.line_numbers.insert("1.0", "1"); self.line_numbers.configure(state="disabled")
        
        self.textbox = ctk.CTkTextbox(self, font=code_font, wrap="none")
        self.textbox.grid(row=0, column=1, sticky="nsew")
        
        tab_width = code_font.measure("    ")
        self.textbox.configure(tabs=tab_width)

        self.tag_colors = {Token.Keyword: "#CC7A00", Token.Name.Function: "#00519E", Token.Name.Class: "#00519E", Token.String: "#008000", Token.Number: "#D85C6B", Token.Comment: "#9D9D9D", Token.Operator: "#CC7A00"}
        for token, color in self.tag_colors.items(): self.textbox.tag_config(str(token), foreground=color)
        
        self.textbox.bind("<KeyRelease>", self.on_key_release); self.textbox.bind("<Return>", self.on_return); self.textbox.bind("<Configure>", lambda e: self.update_line_numbers())
        self.textbox.bind("<MouseWheel>", self.on_mouse_wheel)
        self.textbox.bind("<Button-4>", self.on_mouse_wheel); self.textbox.bind("<Button-5>", self.on_mouse_wheel)
        self.textbox._textbox.bind("<MouseWheel>", self.on_mouse_wheel)
        self.textbox._textbox.bind("<Button-4>", self.on_mouse_wheel); self.textbox._textbox.bind("<Button-5>", self.on_mouse_wheel)
        self.textbox._textbox.configure(yscrollcommand=self.sync_scroll); self.line_numbers._textbox.configure(yscrollcommand=self.sync_scroll)
    
    def on_mouse_wheel(self, event):
        if event.delta: delta = -2 if event.delta > 0 else 2
        else: delta = -2 if event.num == 4 else 2
        self.textbox._textbox.yview_scroll(delta, "units")
        self.line_numbers._textbox.yview_moveto(self.textbox._textbox.yview()[0])
        return "break"
    
    def sync_scroll(self, *args): self.textbox._textbox.yview_moveto(args[0]); self.line_numbers._textbox.yview_moveto(args[0]); return "break"
    def on_return(self, event): self.textbox.insert(ctk.INSERT, "\n"); self.update_syntax_highlighting(); return "break"
    def on_key_release(self, event=None): 
        cursor_pos = self.textbox.index(ctk.INSERT); view_pos = self.textbox._textbox.yview(); line_numbers_view = self.line_numbers._textbox.yview()
        self.update_syntax_highlighting()
        self.textbox.mark_set(ctk.INSERT, cursor_pos); self.textbox._textbox.yview_moveto(view_pos[0]); self.line_numbers._textbox.yview_moveto(line_numbers_view[0])
        
    def update_line_numbers(self):
        current_view = self.line_numbers._textbox.yview()
        self.line_numbers.configure(state="normal"); self.line_numbers.delete("1.0", "end")
        try:
            line_count = int(self.textbox.index("end-1c").split('.')[0])
        except (ValueError, IndexError):
            line_count = 1
        line_numbers_string = "\n".join(str(i) for i in range(1, line_count + 1))
        self.line_numbers.insert("1.0", line_numbers_string); self.line_numbers.configure(state="disabled")
        self.line_numbers._textbox.yview_moveto(current_view[0])
        
    def update_syntax_highlighting(self, event=None):
        cursor_pos = self.textbox.index(ctk.INSERT); textbox_view = self.textbox._textbox.yview()
        for tag in self.tag_colors.keys(): self.textbox.tag_remove(str(tag), "1.0", "end")
        text = self.textbox.get("1.0", "end-1c")
        if not text: self.update_line_numbers(); return
        start_pos = "1.0"
        for token, content in lex(text, self.language_lexer):
            end_pos = f"{start_pos}+{len(content)}c"; base_token = token
            while base_token not in self.tag_colors and base_token.parent: base_token = base_token.parent
            if base_token in self.tag_colors: self.textbox.tag_add(str(base_token), start_pos, end_pos)
            start_pos = end_pos
        self.update_line_numbers()
        self.textbox.mark_set(ctk.INSERT, cursor_pos); self.textbox._textbox.yview_moveto(textbox_view[0])
        
    def insert(self, index, text): self.textbox.insert(index, text); self.update_syntax_highlighting()
    def get(self, start, end): return self.textbox.get(start, end)

# --- Reusable UI Components ---
class ListEditorFrame(ctk.CTkFrame):
    def __init__(self, master, title, initial_list):
        super().__init__(master, fg_color="transparent"); self.items = list(initial_list)
        ctk.CTkLabel(self, text=title, font=ctk.CTkFont(weight="bold")).pack(fill="x", padx=5, pady=2)
        self.entries_frame = ctk.CTkFrame(self, fg_color="transparent"); self.entries_frame.pack(fill="x", expand=True)
        ctk.CTkButton(self, text="+ Add", width=80, command=self.add_item).pack(pady=5)
        self.refresh()
    def add_item(self, value=""): self.items.append(value); self.refresh()
    def remove_item(self, index): self.items.pop(index); self.refresh()
    def get_data(self): return [widget.get() for frame in self.entries_frame.winfo_children() if isinstance(frame, ctk.CTkFrame) for widget in frame.winfo_children() if isinstance(widget, ctk.CTkEntry)]
    def refresh(self):
        for widget in self.entries_frame.winfo_children(): widget.destroy()
        for i, item in enumerate(self.items):
            row_frame = ctk.CTkFrame(self.entries_frame, fg_color="transparent"); row_frame.pack(fill="x", pady=2)
            entry = ctk.CTkEntry(row_frame); entry.insert(0, item); entry.pack(side="left", fill="x", expand=True, padx=5)
            remove_btn = ctk.CTkButton(row_frame, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda index=i: self.remove_item(index)); remove_btn.pack(side="left", padx=5)

# --- Base class for all agent editor panels ---
class BaseEditorFrame(ctk.CTkFrame):
    def __init__(self, master, agent_name, agent_data, app_ref):
        super().__init__(master)
        self.agent_name = agent_name
        self.data = agent_data
        self.app_ref = app_ref
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def get_data(self): raise NotImplementedError

    def _create_help_button(self, parent, help_text):
        def show_help(): self.app_ref.show_help_modal("Help", help_text)
        return ctk.CTkButton(parent, text="?", width=25, height=25, command=show_help)

    def _get_io_help_text(self):
        return """
Inputs, Outputs, and Optional Inputs define the 'signature' of your agent, similar to a function in a programming language.

- Inputs: These are required parameters. If an agent is used as a step in a workflow, these parameters must be provided.

- Optional Inputs: These are parameters that are not required. Your agent's logic should be able to handle cases where these are not provided.

- Outputs: These are the names of the variables that your agent will produce and add to the results 'tape' for subsequent steps to use.
"""

# --- Editors for "proc" and "template" ---
class ProcEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref):
        super().__init__(master, agent_name, agent_data, app_ref)
        tab_view = ctk.CTkTabview(self); tab_view.grid(row=0, column=0, sticky="nsew")
        self.create_settings_tab(tab_view.add("Settings"))
        self.create_inputs_tab(tab_view.add("Inputs"))
        self.create_optionals_tab(tab_view.add("Optional Inputs"))
        self.create_outputs_tab(tab_view.add("Outputs"))
        self.create_function_tab(tab_view.add("Function"))
    def create_settings_tab(self, tab):
        help_btn = self._create_help_button(tab, "Settings for this proc agent.\n\n- Agent Name: Unique identifier.\n- Help Text: Description of function.\n- GUI Settings: Opens an advanced editor for editor-time behaviors like auto-wiring and visual indentation.")
        help_btn.place(relx=0.98, rely=0.02, anchor="ne")
        ctk.CTkLabel(tab, text="Agent Name:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.name_entry = ctk.CTkEntry(tab); self.name_entry.insert(0, self.agent_name); self.name_entry.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(tab, text="Help Text:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.help_text = ctk.CTkTextbox(tab, height=200); self.help_text.insert("1.0", self.data.get("help", "")); self.help_text.pack(fill="x", padx=10, pady=5)
        ctk.CTkButton(tab, text="Advanced GUI Settings...", command=self.open_gui_settings).pack(anchor="w", padx=10, pady=10)
    def create_inputs_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.inputs_frame = ListEditorFrame(tab, "Required Inputs", self.data.get("inputs", [])); self.inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
    def create_optionals_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.optionals_frame = ListEditorFrame(tab, "Optional Inputs", self.data.get("optional_inputs", [])); self.optionals_frame.pack(fill="both", expand=True, padx=5, pady=5)
    def create_outputs_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.outputs_frame = ListEditorFrame(tab, "Outputs", self.data.get("outputs", [])); self.outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
    def create_function_tab(self, tab):
        tab.grid_columnconfigure(0, weight=1); tab.grid_rowconfigure(3, weight=1)
        help_btn = self._create_help_button(tab, "Define the core logic of this proc agent.\n\n- Function Name: The name of the Python function.\n- Function Definition: The Python code to be executed by the core engine. It must match the function name and handle the defined inputs/outputs.")
        help_btn.place(relx=0.98, rely=0.02, anchor="ne")
        ctk.CTkLabel(tab, text="Function Name").grid(row=0, column=0, sticky="w", padx=5, pady=(10,0))
        self.func_name_entry = ctk.CTkEntry(tab); self.func_name_entry.insert(0, self.data.get("function", "")); self.func_name_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(tab, text="Function Definition").grid(row=2, column=0, sticky="w", padx=5, pady=(10,0))
        self.func_def_text = CTkCodeEditor(tab); self.func_def_text.insert("1.0", self.data.get("function_def", "")); self.func_def_text.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    def open_gui_settings(self):
        current_gui_data = self.data.get("gui", {}); modal = GuiSettingsModal(self, current_gui_data); self.wait_window(modal)
        if modal.saved:
            updated_gui_data = modal.get_result()
            if updated_gui_data: self.data["gui"] = updated_gui_data
            elif "gui" in self.data: del self.data["gui"]

    def get_data(self):
        updated_data = copy.deepcopy(self.data)
        updated_data['name'] = self.name_entry.get().strip()
        updated_data['help'] = self.help_text.get("1.0", "end-1c").strip()
        updated_data['inputs'] = self.inputs_frame.get_data()
        updated_data['optional_inputs'] = self.optionals_frame.get_data()
        updated_data['outputs'] = self.outputs_frame.get_data()
        updated_data['function'] = self.func_name_entry.get()
        updated_data['function_def'] = self.func_def_text.get("1.0", "end-1c").strip()
        return updated_data

class TemplateEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref):
        super().__init__(master, agent_name, agent_data, app_ref)
        tab_view = ctk.CTkTabview(self); tab_view.grid(row=0, column=0, sticky="nsew")
        self.create_settings_tab(tab_view.add("Settings"))
        self.create_inputs_tab(tab_view.add("Inputs"))
        self.create_optionals_tab(tab_view.add("Optional Inputs"))
        self.create_outputs_tab(tab_view.add("Outputs"))
        self.create_prompt_tab(tab_view.add("Prompt"))
    def create_settings_tab(self, tab):
        help_btn = self._create_help_button(tab, "Settings for this template agent.\n\n- Agent Name: Unique identifier.\n- Help Text: Description of the template's purpose.")
        help_btn.place(relx=0.98, rely=0.02, anchor="ne")
        ctk.CTkLabel(tab, text="Agent Name:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.name_entry = ctk.CTkEntry(tab); self.name_entry.insert(0, self.agent_name); self.name_entry.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(tab, text="Help Text:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.help_text = ctk.CTkTextbox(tab, height=200); self.help_text.insert("1.0", self.data.get("help", "")); self.help_text.pack(fill="x", padx=10, pady=5)
    def create_inputs_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.inputs_frame = ListEditorFrame(tab, "Required Inputs", self.data.get("inputs", [])); self.inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
    def create_optionals_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.optionals_frame = ListEditorFrame(tab, "Optional Inputs", self.data.get("optional_inputs", [])); self.optionals_frame.pack(fill="both", expand=True, padx=5, pady=5)
    def create_outputs_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.outputs_frame = ListEditorFrame(tab, "Outputs", self.data.get("outputs", [])); self.outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
    def create_prompt_tab(self, tab):
        tab.grid_rowconfigure(1, weight=1); tab.grid_columnconfigure(0, weight=1)
        help_btn = self._create_help_button(tab, "Define the template text.\n\nUse curly braces {variable_name} for placeholders that will be replaced with values from the Inputs tab.")
        help_btn.place(relx=0.98, rely=0.02, anchor="ne")
        ctk.CTkLabel(tab, text="Prompt Template").grid(row=0, column=0, pady=(5,0))
        self.prompt_text = ctk.CTkTextbox(tab); self.prompt_text.insert("1.0", self.data.get("prompt", ""))
        self.prompt_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def get_data(self):
        updated_data = copy.deepcopy(self.data)
        updated_data['name'] = self.name_entry.get().strip()
        updated_data['help'] = self.help_text.get("1.0", "end-1c").strip()
        updated_data['inputs'] = self.inputs_frame.get_data()
        updated_data['optional_inputs'] = self.optionals_frame.get_data()
        updated_data['outputs'] = self.outputs_frame.get_data()
        updated_data['prompt'] = self.prompt_text.get("1.0", "end-1c").strip()
        return updated_data

class JsonEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref):
        super().__init__(master, agent_name, agent_data, app_ref)
        ctk.CTkLabel(self, text="Raw JSON Editor", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", padx=10, pady=(10,0))
        self.textbox = ctk.CTkTextbox(self, font=("monospace", 12)); self.textbox.pack(fill="both", expand=True, padx=10, pady=10)
        self.textbox.insert("1.0", json.dumps(self.data, indent=2))
        
    def get_data(self):
        try:
            data = json.loads(self.textbox.get("1.0", "end-1c"))
            data['name'] = self.agent_name
            if 'type' not in data and 'type' in self.data:
                data['type'] = self.data['type']
            return data
        except json.JSONDecodeError as e:
            messagebox.showerror("JSON Error", f"Invalid JSON: {e}")
            return None

# --- Global Config Editor Modal ---
class GlobalConfigEditorModal(ctk.CTkToplevel):
    def __init__(self, parent, config_manager):
        super().__init__(parent); self.title("Global Configuration Editor"); self.geometry("800x600")
        self.config_manager = config_manager; self.saved = False
        self.config_data = copy.deepcopy(config_manager.config)
        if "agents" in self.config_data: del self.config_data["agents"]
        self.create_widgets(); self.transient(parent); self.grab_set()
    def create_widgets(self):
        content_frame = ctk.CTkFrame(self); content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        ctk.CTkLabel(content_frame, text="Global Configuration (JSON)", font=ctk.CTkFont(size=16, weight="bold")).pack(pady=(0, 10))
        warning_label = ctk.CTkLabel(content_frame, text="⚠️ Warning: This editor modifies all configuration except agents. Edit carefully!", text_color="orange"); warning_label.pack(pady=(0, 10))
        self.json_textbox = ctk.CTkTextbox(content_frame, font=("monospace", 12)); self.json_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.json_textbox.insert("1.0", json.dumps(self.config_data, indent=2))
        button_frame = ctk.CTkFrame(self); button_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save Configuration", command=self.save, fg_color="green").pack(side="right", padx=10)
    def save(self):
        try:
            new_config = json.loads(self.json_textbox.get("1.0", "end-1c"))
            if "agents" in new_config: messagebox.showerror("Error", "Cannot modify 'agents' key through this editor!"); return
            agents_backup = self.config_manager.config.get("agents", {}); self.config_manager.config = new_config
            self.config_manager.config["agents"] = agents_backup; self.saved = True; self.destroy()
        except json.JSONDecodeError as e: messagebox.showerror("JSON Error", f"Invalid JSON: {e}")
        except Exception as e: messagebox.showerror("Error", f"Failed to save configuration: {e}")
    def cancel(self): self.saved = False; self.destroy()

# --- Main Application Window ---
class App(ctk.CTk):
    # MODIFIED to accept config_path from main block
    def __init__(self, config_path="config.json"):
        super().__init__()
        self.title("Agent Workflow Editor")
        self.geometry("1400x800")
        # MODIFIED to use the passed-in config_path
        self.config_manager = ConfigManager(config_path=config_path)
        self.current_agent_name = None
        self.editor_frame_instance = None
        self.search_text = ctk.StringVar()
        self.search_text.trace("w", self.on_search_changed)
        self.show_hidden_agents_var = ctk.IntVar(value=0)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)

        self.create_agent_list_panel()
        self.create_editor_panel()
        self.create_modal_overlay()
        self.refresh_agent_list()
        self.show_welcome_message()

    def create_agent_list_panel(self):
        self.agent_list_frame = ctk.CTkFrame(self, width=320)
        frame_bg_color = self.agent_list_frame.cget("fg_color")
        self.agent_list_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="ns")
        self.agent_list_frame.grid_propagate(False)
        self.agent_list_frame.grid_rowconfigure(2, weight=1); self.agent_list_frame.grid_columnconfigure(0, weight=1)
        header_frame = ctk.CTkFrame(self.agent_list_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew"); header_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(header_frame, text="Agents", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, sticky="w")
        ctk.CTkCheckBox(header_frame, text="", variable=self.show_hidden_agents_var, command=self.refresh_agent_list, width=20, fg_color=frame_bg_color, hover_color=frame_bg_color, border_width=0).grid(row=0, column=1, padx=(0,0))
        
        add_menu = ctk.CTkOptionMenu(header_frame, width=120, values=["Workflow", "Proc", "Template"], command=self.add_new_agent)
        add_menu.grid(row=0, column=2, padx=(0, 5)); add_menu.set("Create New...")
        config_btn = ctk.CTkButton(header_frame, text="⚙️", width=30, command=self.open_global_config)
        config_btn.grid(row=0, column=3)
        search_frame = ctk.CTkFrame(self.agent_list_frame, fg_color="transparent")
        search_frame.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew"); search_frame.grid_columnconfigure(0, weight=1)
        self.search_entry = ctk.CTkEntry(search_frame, textvariable=self.search_text, placeholder_text="Filter agents...")
        self.search_entry.grid(row=0, column=0, sticky="ew")
        clear_search_btn = ctk.CTkButton(search_frame, text="X", width=30, text_color="white", fg_color="#D32F2F", hover_color="#B71C1C", command=lambda: self.search_text.set(""))
        clear_search_btn.grid(row=0, column=1, padx=(5, 0))
        self.agent_scroll_frame = ctk.CTkScrollableFrame(self.agent_list_frame)
        self.agent_scroll_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")
        
    def create_editor_panel(self):
        self.editor_container = ctk.CTkFrame(self); self.editor_container.grid(row=0, column=1, padx=10, pady=(10,0), sticky="nsew")
        self.editor_container.grid_rowconfigure(0, weight=1); self.editor_container.grid_columnconfigure(0, weight=1)
        
        self.action_bar = ctk.CTkFrame(self, fg_color="transparent"); self.action_bar.grid(row=1, column=1, padx=10, pady=(5, 10), sticky="ew")
        
        self.save_btn = ctk.CTkButton(self.action_bar, text="Save Agent", command=self.save_agent, fg_color="green")
        self.run_btn = ctk.CTkButton(self.action_bar, text="▶️ Run", command=self.open_run_modal, fg_color="#FFC700", text_color="#000000", hover_color="#FFA000")
        
        # This is the original, correct logic for disabling the button.
        if dynamic_workflows_agents is None:
            self.run_btn.configure(state="disabled")

        self.cancel_btn = ctk.CTkButton(self.action_bar, text="Cancel", command=self.show_welcome_message, fg_color="#D32F2F", hover_color="#B71C1C")
        self.spacer_frame = ctk.CTkFrame(self.action_bar, fg_color="transparent", width=240,height=40)

    def create_modal_overlay(self):
        self.overlay = ctk.CTkFrame(self, fg_color=("#000000", "#000000")); self.overlay.lower()
        self.overlay_label = ctk.CTkLabel(self.overlay, text="", font=ctk.CTkFont(size=24, weight="bold"))
        
    def show_overlay(self, text="Editing Step...\nMain window is locked."):
        self.overlay_label.configure(text=text)
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1); self.overlay_label.place(relx=0.5, rely=0.5, anchor="center"); self.overlay.lift()
        
    def hide_overlay(self): self.overlay.place_forget()
    
    def on_search_changed(self, *args):
        self.refresh_agent_list()
    
    def refresh_agent_list(self):
        for widget in self.agent_scroll_frame.winfo_children(): widget.destroy()
        search_term = self.search_text.get().lower()
        show_hidden = self.show_hidden_agents_var.get() == 1
        
        is_workflow_steps_tab = (isinstance(self.editor_frame_instance, WorkflowEditorFrame) and
                                 self.editor_frame_instance.tab_view.get() == "Steps")
        
        for name in self.config_manager.get_agent_names():
            agent_data = self.config_manager.get_agent_data(name)
            if agent_data.get("gui", {}).get("hide_in_agent_list", False) and not show_hidden: continue
            if search_term and search_term not in name.lower(): continue
            
            agent_type = agent_data.get("type", "json")
            prefix = {"workflow": "W", "proc": "P", "template": "T"}.get(agent_type, "J")
            
            row = ctk.CTkFrame(self.agent_scroll_frame, fg_color="transparent")
            row.pack(fill="x", padx=2, pady=2)
            
            del_btn = ctk.CTkButton(row, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda n=name: self.delete_agent(agent_name=n, confirm=True))
            del_btn.pack(side="right")
            
            btn = ctk.CTkButton(row, text=f"[{prefix}] {name}", anchor="w")
            btn.pack(side="left", fill="x", expand=True)

            if is_workflow_steps_tab:
                workflow_editor = self.editor_frame_instance
                btn.bind("<ButtonPress-1>", lambda e, n=name: workflow_editor._on_agent_drag_start(e, n))
            else:
                btn.configure(command=lambda n=name: self.on_agent_list_click(n))

    def on_agent_list_click(self, agent_name):
        self.select_agent(agent_name)

    def add_new_agent(self, choice):
        agent_type = choice.lower(); new_name = self.config_manager.create_new_agent(agent_type)
        self.config_manager.save(); self.refresh_agent_list(); self.select_agent(new_name)
    def select_agent(self, agent_name): self.current_agent_name = agent_name; self.build_editor_form()
    
    def show_welcome_message(self):
        if self.editor_frame_instance: self.editor_frame_instance.destroy()
        self.current_agent_name = None; self.editor_frame_instance = None
        label = ctk.CTkLabel(self.editor_container, text="Select an agent to edit or create a new one.", font=ctk.CTkFont(size=24)); label.place(relx=0.5, rely=0.5, anchor="center")
        self.action_bar.grid_remove()
        self.refresh_agent_list()
        
    def build_editor_form(self):
        if self.editor_frame_instance: self.editor_frame_instance.destroy()
        self.action_bar.grid()
        
        for widget in [self.save_btn, self.run_btn, self.cancel_btn, self.spacer_frame]: widget.pack_forget()
        self.save_btn.pack(side="right", padx=10, pady=5)
        self.run_btn.pack(side="right", padx=0, pady=5)
        self.spacer_frame.pack(side="right")
        self.cancel_btn.pack(side="right", padx=0, pady=5)

        agent_data = self.config_manager.get_agent_data(self.current_agent_name)
        editor_class = {"workflow": WorkflowEditorFrame, "proc": ProcEditorFrame, "template": TemplateEditorFrame}.get(agent_data.get("type"), JsonEditorFrame)
        self.editor_frame_instance = editor_class(self.editor_container, self.current_agent_name, copy.deepcopy(agent_data), self)
        self.editor_frame_instance.grid(row=0, column=0, sticky="nsew")
        
        self.refresh_agent_list()
        
    def save_agent(self):
        if not self.editor_frame_instance or not self.current_agent_name: return
        updated_data = self.editor_frame_instance.get_data();
        if updated_data is None: return
        
        new_name = updated_data.pop('name', self.current_agent_name)
        if not new_name: messagebox.showerror("Error", "Agent name cannot be empty."); return
        if new_name != self.current_agent_name:
            if not self.config_manager.rename_agent(self.current_agent_name, new_name): messagebox.showerror("Error", f"Agent name '{new_name}' already exists."); return
            self.current_agent_name = new_name
        self.config_manager.update_agent(self.current_agent_name, updated_data); self.config_manager.save(); self.show_toast(f"Agent '{self.current_agent_name}' saved.")
        self.show_welcome_message()
        
    def delete_agent(self, agent_name=None, confirm=False):
        name_to_delete = agent_name;
        if not name_to_delete: return
        deps = self.config_manager.check_agent_usage(name_to_delete)
        if deps: messagebox.showerror("Cannot Delete", f"'{name_to_delete}' is used by:\n- " + "\n- ".join(deps)); return
        if confirm and not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{name_to_delete}'? This cannot be undone."): return
        self.config_manager.delete_agent(name_to_delete); self.config_manager.save(); self.refresh_agent_list()
        if name_to_delete == self.current_agent_name: self.show_welcome_message()
        
    def open_global_config(self):
        modal = GlobalConfigEditorModal(self, self.config_manager); self.wait_window(modal)
        if modal.saved: self.config_manager.save(); self.show_toast("Global configuration saved successfully!")
    
    def open_step_editor(self, index):
        if not isinstance(self.editor_frame_instance, WorkflowEditorFrame): return

        parent_workflow_data = self.editor_frame_instance.get_data()
        if parent_workflow_data is None:
            messagebox.showerror("Error", "Could not read current workflow data. Please check for errors.")
            return

        step_data = parent_workflow_data["steps"][index]
        agent_name = step_data.get("agent", "")
        agent_def = self.config_manager.get_agent_data(agent_name)

        if not agent_def:
            messagebox.showerror("Agent Not Found", f"The agent '{agent_name}' used in step {index} could not be found.")
            return

        self.show_overlay()
        modal = StepEditorModal(self, index, step_data, agent_def, parent_workflow_data)
        self.wait_window(modal)
        self.hide_overlay()
        
        if modal.saved:
            self.editor_frame_instance.data["steps"][index] = modal.get_result()
            self.editor_frame_instance.refresh_steps_list()
            
    def open_run_modal(self):
        if not self.editor_frame_instance: return
            
        current_agent_data = self.editor_frame_instance.get_data()
        if current_agent_data is None:
            messagebox.showerror("Error", "Cannot run agent. Please check editor for errors (e.g., invalid JSON).")
            return

        self.show_overlay(f"Running '{self.current_agent_name}'...\nMain window is locked.")
        modal = RunAgentModal(self, self.current_agent_name, current_agent_data)
        self.wait_window(modal)
        self.hide_overlay()

    def show_toast(self, message):
        toast = ctk.CTkLabel(self, text=message, fg_color=("#333", "#555"), text_color="white", corner_radius=10, font=("", 14))
        toast.place(relx=0.5, rely=0.95, anchor="center"); toast.lift(); self.after(2500, toast.destroy)
        
    def show_help_modal(self, title, content):
        help_window = ctk.CTkToplevel(self)
        help_window.title(title); help_window.geometry("600x600")
        help_window.transient(self); help_window.grab_set()
        textbox = ctk.CTkTextbox(help_window, wrap="word", font=("", 14))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)
        textbox.insert("1.0", content); textbox.configure(state="disabled")

# --- Custom JSON Encoder for Run Modal ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            try: return o.decode('utf-8')
            except UnicodeDecodeError: return f"<base64_encoded_bytes>{base64.b64encode(o).decode('utf-8')}</base64_encoded_bytes>"
        return json.JSONEncoder.default(self, o)

# --- Run Agent Modal Window ---
class RunAgentModal(ctk.CTkToplevel):
    def __init__(self, parent_app, agent_name, agent_data):
        super().__init__(parent_app)
        self.title(f"Run Agent: {agent_name}")
        self.geometry("900x700")
        self.app = parent_app
        self.agent_name = agent_name
        self.agent_data = agent_data
        self.input_entries = {}
        self.optional_frames = {}
        self.test_widgets = []
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        self.create_widgets()
        self.load_last_run_config()
        self.transient(parent_app)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def close(self):
        self.save_current_run_config()
        self.destroy()

    def create_widgets(self):
        input_container = ctk.CTkFrame(self)
        input_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        input_container.grid_columnconfigure(0, weight=1)
        self.inputs_scroll_frame = ctk.CTkScrollableFrame(input_container, label_text="Inputs")
        self.inputs_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        button_frame = ctk.CTkFrame(input_container, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(5,10))
        ctk.CTkLabel(button_frame, text="Log Level:").pack(side="left", padx=(0,5))
        self.log_level_var = ctk.StringVar(value="INFO")
        log_level_menu = ctk.CTkOptionMenu(button_frame, variable=self.log_level_var, values=["DEBUG", "INFO", "WARNING", "ERROR"])
        log_level_menu.pack(side="left")
        help_button = ctk.CTkButton(button_frame, text="?", width=25, command=self.show_test_help)
        help_button.pack(side="left", padx=10)
        run_button = ctk.CTkButton(button_frame, text="Run Agent & Validate", command=self.execute_run, fg_color="green")
        run_button.pack(side="right")
        output_container = ctk.CTkFrame(self)
        output_container.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        output_container.grid_columnconfigure(0, weight=1)
        output_container.grid_rowconfigure(0, weight=1)
        self.tab_view = ctk.CTkTabview(output_container)
        self.tab_view.pack(fill="both", expand=True)
        self.output_tab = self.tab_view.add("Final Output")
        self.log_tab = self.tab_view.add("Verbose Log")
        self.test_tab = self.tab_view.add("Unit Tests")
        self.output_textbox = ctk.CTkTextbox(self.output_tab, font=("monospace", 12), wrap="word")
        self.output_textbox.pack(fill="both", expand=True)
        self.log_textbox = ctk.CTkTextbox(self.log_tab, font=("monospace", 12), wrap="word")
        self.log_textbox.pack(fill="both", expand=True)
        self.create_test_tab_widgets()
        self.populate_input_fields()
        
    def create_test_tab_widgets(self):
        self.test_tab.grid_columnconfigure(0, weight=1)
        self.test_tab.grid_rowconfigure(1, weight=1)
        top_frame = ctk.CTkFrame(self.test_tab)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.test_summary_label = ctk.CTkLabel(top_frame, text="Run tests to see results.", font=ctk.CTkFont(weight="bold"))
        self.test_summary_label.pack(side="left")
        ctk.CTkButton(top_frame, text="+ Add Test Case", command=self.add_test_case).pack(side="right")
        self.test_cases_frame = ctk.CTkScrollableFrame(self.test_tab, label_text="Assertions")
        self.test_cases_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def populate_input_fields(self):
        for widget in self.inputs_scroll_frame.winfo_children(): widget.destroy()
        self.input_entries.clear()
        required_inputs = self.agent_data.get("inputs", [])
        if required_inputs:
            ctk.CTkLabel(self.inputs_scroll_frame, text="Required", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
            for key in required_inputs: self.add_input_field(self.inputs_scroll_frame, key, is_optional=False)
        all_optionals = self.agent_data.get("optional_inputs", [])
        if all_optionals:
            ctk.CTkLabel(self.inputs_scroll_frame, text="Optional", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(10,0))
            self.optional_menu = ctk.CTkOptionMenu(self.inputs_scroll_frame, values=["Add Optional Input..."] + sorted(all_optionals), command=self.add_optional_from_menu)
            self.optional_menu.pack(anchor="w", padx=5, pady=5)

    def add_input_field(self, parent, key, is_optional=True):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", expand=True, pady=2)
        ctk.CTkLabel(frame, text=key, width=150).pack(side="left", padx=5)
        entry = ctk.CTkEntry(frame); entry.pack(side="left", fill="x", expand=True, padx=5)
        self.input_entries[key] = entry
        if is_optional:
            remove_btn = ctk.CTkButton(frame, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda k=key: self.remove_optional_field(k))
            remove_btn.pack(side="left", padx=5)
            self.optional_frames[key] = frame

    def add_optional_from_menu(self, choice):
        if "Add Optional Input..." not in choice and choice not in self.optional_frames:
            self.add_input_field(self.inputs_scroll_frame, choice, is_optional=True)
            current_values = self.optional_menu.cget("values")
            current_values.remove(choice)
            self.optional_menu.configure(values=current_values if len(current_values) > 1 else ["No more optionals"])
            self.optional_menu.set("Add Optional Input...")

    def remove_optional_field(self, key):
        if key in self.optional_frames:
            self.optional_frames[key].destroy()
            del self.optional_frames[key]
            del self.input_entries[key]
            current_values = self.optional_menu.cget("values")
            if "No more optionals" in current_values: current_values.remove("No more optionals")
            if key not in current_values: current_values.append(key)
            self.optional_menu.configure(values=sorted(current_values))
    
    def add_test_case(self, test_data=None):
        if test_data is None: test_data = { "output_variable": "", "assertion_type": "Equals", "expected_value": "" }
        frame = ctk.CTkFrame(self.test_cases_frame); frame.pack(fill="x", pady=2)
        frame.grid_columnconfigure(2, weight=1)
        output_vars = self.agent_data.get("outputs", []) + ["status.value"]
        var_menu = ctk.CTkOptionMenu(frame, values=sorted(output_vars)); var_menu.grid(row=0, column=0, padx=5, pady=5)
        var_menu.set(test_data.get("output_variable") or "Select Variable")
        assertion_menu = ctk.CTkOptionMenu(frame, values=["Equals", "Regex Match"]); assertion_menu.grid(row=0, column=1, padx=5, pady=5)
        assertion_menu.set(test_data.get("assertion_type", "Equals"))
        value_entry = ctk.CTkEntry(frame, placeholder_text="Expected Value"); value_entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        value_entry.insert(0, test_data.get("expected_value", ""))
        result_label = ctk.CTkLabel(frame, text="?", width=60, font=ctk.CTkFont(weight="bold")); result_label.grid(row=0, column=3, padx=5, pady=5)
        remove_btn = ctk.CTkButton(frame, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda f=frame: f.destroy())
        remove_btn.grid(row=0, column=4, padx=5, pady=5)
        self.test_widgets.append({ "frame": frame, "var_menu": var_menu, "assertion_menu": assertion_menu, "value_entry": value_entry, "result_label": result_label })
        
    def load_last_run_config(self):
        run_config = self.agent_data.get("run_config", {})
        last_inputs = run_config.get("last_inputs", {})
        self.log_level_var.set(run_config.get("last_log_level", "INFO"))
        for key, value in last_inputs.items():
            if key in self.input_entries: self.input_entries[key].insert(0, value)
            elif key in self.agent_data.get("optional_inputs", []) and key not in self.optional_frames:
                self.add_optional_from_menu(key)
                if key in self.input_entries: self.input_entries[key].insert(0, value)
        for test in run_config.get("tests", []): self.add_test_case(test)
            
    def save_current_run_config(self):
        workflow_inputs = {key: entry.get() for key, entry in self.input_entries.items()}
        tests_data = []
        for widget_set in self.test_widgets:
            if widget_set["frame"].winfo_exists():
                tests_data.append({"output_variable": widget_set["var_menu"].get(), "assertion_type": widget_set["assertion_menu"].get(), "expected_value": widget_set["value_entry"].get()})
        new_run_config = {"last_inputs": workflow_inputs, "last_log_level": self.log_level_var.get(), "tests": tests_data}
        if self.app.editor_frame_instance and hasattr(self.app.editor_frame_instance, 'data'):
            if workflow_inputs or tests_data:
                self.app.editor_frame_instance.data['run_config'] = new_run_config

    def execute_run(self):
        self.save_current_run_config()
        if dynamic_workflows_agents:
            dynamic_workflows_agents.log_text_limit = int(self.app.config_manager.config.get('workflow_settings', {}).get('log_text_limit', 500))
        workflow_inputs = {key: entry.get() for key, entry in self.input_entries.items()}
        temp_config = copy.deepcopy(self.app.config_manager.config)
        temp_config['agents'][self.agent_name] = self.agent_data
        setup_depth_manager(temp_config)
        agent_to_run = self.agent_data
        if not agent_to_run: messagebox.showerror("Error", "Could not prepare agent for execution."); return
        log_stream = io.StringIO()
        ui_log_handler = logging.StreamHandler(log_stream)
        ui_log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
        root_logger.addHandler(ui_log_handler)
        root_logger.setLevel(getattr(logging, self.log_level_var.get(), logging.INFO))
        final_result_tape, final_status = {}, {"status": {"value": -99, "reason": "Execution did not run"}}
        try:
            final_result_tape, final_status = exec_agent(agent=agent_to_run, agent_name=self.agent_name, config=temp_config, cli_args=workflow_inputs, results={})
        except Exception as e:
            final_result_tape = {"__error__": "An unhandled exception occurred during workflow execution.", "details": str(e)}
            logging.exception("Workflow execution failed")
        for textbox in [self.log_textbox, self.output_textbox]: textbox.configure(state="normal"); textbox.delete("1.0", "end")
        self.log_textbox.insert("1.0", log_stream.getvalue())
        self.output_textbox.insert("1.0", json.dumps(final_result_tape, indent=2, cls=CustomJSONEncoder))
        for textbox in [self.log_textbox, self.output_textbox]: textbox.configure(state="disabled")
        self.run_assertions(final_result_tape, final_status)
        self.tab_view.set("Unit Tests")

    def run_assertions(self, final_result_tape, final_status):
        active_widgets = [w for w in self.test_widgets if w["frame"].winfo_exists()]
        if not active_widgets: self.test_summary_label.configure(text="No tests defined."); return
        passed_count = 0
        for widget_set in active_widgets:
            test = {"output_variable": widget_set["var_menu"].get(), "assertion_type": widget_set["assertion_menu"].get(), "expected_value": widget_set["value_entry"].get()}
            actual_value = final_status.get('status', {}).get('value') if test["output_variable"] == 'status.value' else get_nested(final_result_tape, test["output_variable"])
            test_passed = False
            try:
                if actual_value is not None:
                    if test["assertion_type"] == "Regex Match" and re.search(str(test["expected_value"]), str(actual_value)): test_passed = True
                    elif test["assertion_type"] == "Equals" and str(actual_value) == str(test["expected_value"]): test_passed = True
            except Exception: test_passed = False
            if test_passed:
                passed_count += 1
                widget_set["result_label"].configure(text="PASS", text_color="green")
            else: widget_set["result_label"].configure(text="FAIL", text_color="red")
        total_tests = len(active_widgets)
        summary_text = f"Results: {passed_count} / {total_tests} tests passed."
        summary_color = "green" if passed_count == total_tests else "red"
        self.test_summary_label.configure(text=summary_text, text_color=summary_color)
        
    def show_test_help(self):
        self.app.show_help_modal("Unit Testing Help", """**Agent Unit Testing** ... """) # Content omitted for brevity

# --- Step Editor Modal Window ---
class StepEditorModal(ctk.CTkToplevel):
    def __init__(self, parent, index, step_data, agent_def, parent_workflow_data=None):
        super().__init__(parent)
        self.title(f"Edit Step {index}: {step_data.get('agent')}")
        self.geometry("1100x700")
        self.agent_def = agent_def or {}
        self.editing_data = copy.deepcopy(step_data)
        self.step_index = index
        self.parent_workflow_data = parent_workflow_data or {"inputs": [], "optional_inputs": [], "steps": []}
        self.active_entry = None
        self.available_vars = set()
        dummy_entry = ctk.CTkEntry(self)
        self.default_border_color = dummy_entry.cget("border_color")
        self.literal_border_color = "#0A477A"
        dummy_entry.destroy()
        self._calculate_available_variables()
        self.saved = False
        self.param_entries = {}
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)
        self.create_widgets()
        self.transient(parent)
        self.grab_set()

    def _calculate_available_variables(self):
        self.available_vars.clear()
        for var in self.parent_workflow_data.get('inputs', []): self.available_vars.add(var)
        for var in self.parent_workflow_data.get('optional_inputs', []): self.available_vars.add(var)
        for i in range(self.step_index):
            step = self.parent_workflow_data['steps'][i]
            for output_var in step.get('output', []): self.available_vars.add(output_var)

    def validate_entry_variables(self, entry_widget):
        text = entry_widget.get()
        found_vars = re.findall(r'\$(\w+)', text)
        if not text: entry_widget.configure(border_color=self.default_border_color, border_width=1); return
        if not found_vars: entry_widget.configure(border_color=self.literal_border_color, border_width=1); return
        is_valid = all(var in self.available_vars for var in found_vars)
        entry_widget.configure(border_color="red" if not is_valid else self.default_border_color, border_width=1)

    def set_active_entry(self, entry_widget): self.active_entry = entry_widget

    def append_variable_to_active_entry(self, var_name):
        if self.active_entry:
            cursor_pos = self.active_entry.index(ctk.INSERT)
            variable_string = f" ${var_name}" if cursor_pos > 0 and self.active_entry.get() and self.active_entry.get()[cursor_pos-1] not in (' ', '(') else f"${var_name}"
            self.active_entry.insert(cursor_pos, variable_string)
            self.active_entry.focus()
            self.validate_entry_variables(self.active_entry)

    def populate_variable_selector(self):
        def create_var_button(parent, var_name):
            btn = ctk.CTkButton(parent, text=var_name, anchor="w", fg_color="gray", command=lambda v=var_name: self.append_variable_to_active_entry(v))
            btn.pack(fill="x", padx=5, pady=2)
        for var in sorted(list(self.parent_workflow_data.get('inputs', []))): create_var_button(self.wf_req_inputs_frame, var)
        for var in sorted(list(self.parent_workflow_data.get('optional_inputs', []))): create_var_button(self.wf_opt_inputs_frame, var)
        for i in range(self.step_index):
            step = self.parent_workflow_data['steps'][i]
            ctk.CTkLabel(self.step_outputs_frame, text=f"Step {i}: {step.get('agent', 'Unknown')}", font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(8, 2), padx=5)
            for output_var in step.get('output', []): create_var_button(self.step_outputs_frame, output_var)

    def create_widgets(self):
        selector_frame = ctk.CTkFrame(self, width=300); selector_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        selector_frame.grid_rowconfigure(0, weight=1); selector_frame.grid_columnconfigure(0, weight=1)
        selector_tabs = ctk.CTkTabview(selector_frame); selector_tabs.grid(row=0, column=0, sticky="nsew")
        self.wf_req_inputs_frame = ctk.CTkScrollableFrame(selector_tabs.add("Workflow Inputs"), label_text="Required"); self.wf_req_inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.wf_opt_inputs_frame = ctk.CTkScrollableFrame(selector_tabs.tab("Workflow Inputs"), label_text="Optional"); self.wf_opt_inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.step_outputs_frame = ctk.CTkScrollableFrame(selector_tabs.add("Step Outputs"), label_text="Available Outputs from Previous Steps"); self.step_outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        form = ctk.CTkFrame(self); form.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        form.grid_rowconfigure(0, weight=1); form.grid_columnconfigure(0, weight=1)
        self.params_frame = ctk.CTkScrollableFrame(form, label_text="Parameters & Outputs"); self.params_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        button_frame = ctk.CTkFrame(self); button_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save", command=self.save, fg_color="green").pack(side="right", padx=10)
        self.populate_variable_selector(); self.refresh_params_form()

    def add_optional_param(self, param_name: str):
        if param_name and "Add Optional" not in param_name:
            self.editing_data.setdefault("params", {})[param_name] = ""
            self.refresh_params_form()
            
    def refresh_params_form(self):
        for widget in self.params_frame.winfo_children(): widget.destroy()
        self.param_entries.clear()
        required_inputs = set(self.agent_def.get("inputs", [])); optional_inputs = set(self.agent_def.get("optional_inputs", []))
        self.editing_data.setdefault('params', {}); self.editing_data.setdefault('output', [])
        ctk.CTkLabel(self.params_frame, text="Outputs", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        for i, item in enumerate(self.editing_data['output']):
            entry = ctk.CTkEntry(self.params_frame); entry.insert(0, item); entry.pack(fill="x", padx=5, pady=2)
            self.param_entries[f'output_{i}'] = entry
        if required_inputs: ctk.CTkLabel(self.params_frame, text="Required Parameters", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(15, 0))
        for key in sorted(list(required_inputs)):
            if key not in self.editing_data["params"]: self.editing_data["params"][key] = f"${key}"
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent"); param_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(param_frame, text=key, width=200).pack(side="left", padx=5)
            value_entry = ctk.CTkEntry(param_frame); value_entry.insert(0, self.editing_data["params"].get(key, ""))
            value_entry.pack(side="left", padx=5, expand=True, fill="x")
            value_entry.bind("<FocusIn>", lambda event, entry=value_entry: self.set_active_entry(entry))
            value_entry.bind("<KeyRelease>", lambda event, entry=value_entry: self.validate_entry_variables(entry))
            self.param_entries[f'param_val_{key}'] = value_entry; self.validate_entry_variables(value_entry)
        optional_and_custom_keys = set(self.editing_data["params"].keys()) - required_inputs
        if optional_and_custom_keys: ctk.CTkLabel(self.params_frame, text="Optional / Custom Parameters", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(15, 0))
        for key in sorted(list(optional_and_custom_keys)):
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent"); param_frame.pack(fill="x", pady=2)
            if key in optional_inputs: ctk.CTkLabel(param_frame, text=key, width=200).pack(side="left", padx=5)
            else:
                key_entry = ctk.CTkEntry(param_frame, width=200); key_entry.insert(0, key); key_entry.pack(side="left", padx=5)
                self.param_entries[f'param_key_{key}'] = key_entry
            value_entry = ctk.CTkEntry(param_frame); value_entry.insert(0, self.editing_data["params"].get(key, ""));
            value_entry.pack(side="left", padx=5, expand=True, fill="x")
            value_entry.bind("<FocusIn>", lambda event, entry=value_entry: self.set_active_entry(entry))
            value_entry.bind("<KeyRelease>", lambda event, entry=value_entry: self.validate_entry_variables(entry))
            self.param_entries[f'param_val_{key}'] = value_entry; self.validate_entry_variables(value_entry)
            remove_btn = ctk.CTkButton(param_frame, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda k=key: self.remove_param(k))
            remove_btn.pack(side="left", padx=5)
        available_options = sorted(list(optional_inputs - set(self.editing_data["params"].keys())))
        if available_options:
            option_menu = ctk.CTkOptionMenu(self.params_frame, values=["Add Optional Input..."] + available_options, command=self.add_optional_param)
            option_menu.pack(pady=10, padx=5, anchor="w")

    def remove_param(self, key):
        if key in self.editing_data.get("params", {}) and key not in set(self.agent_def.get("inputs", [])):
            del self.editing_data["params"][key]
        self.refresh_params_form()
        
    def get_form_data(self):
        self.editing_data['output'] = [self.param_entries[f'output_{i}'].get() for i in range(len(self.editing_data.get('output',[])))]
        new_params = {}
        all_rendered_keys = {k.replace('param_val_', '') for k in self.param_entries if k.startswith('param_val_')}
        for key_ref in all_rendered_keys:
            new_key = self.param_entries[f'param_key_{key_ref}'].get() if f'param_key_{key_ref}' in self.param_entries else key_ref
            if new_key: new_params[new_key] = self.param_entries[f'param_val_{key_ref}'].get()
        self.editing_data['params'] = new_params
        
    def save(self): self.get_form_data(); self.saved = True; self.destroy()
    def cancel(self): self.saved = False; self.destroy()
    def get_result(self): return self.editing_data

# --- GUI Settings Editor Modal for Agents ---
class GuiSettingsModal(ctk.CTkToplevel):
    def __init__(self, parent, gui_data):
        super().__init__(parent); self.title("Advanced GUI Settings"); self.geometry("800x700")
        self.gui_data = copy.deepcopy(gui_data) or {}; self.saved = False; self.script_widgets = []
        self.create_widgets(); self.transient(parent); self.grab_set()
    def create_widgets(self):
        self.grid_rowconfigure(0, weight=1); self.grid_columnconfigure(0, weight=1)
        tab_view = ctk.CTkTabview(self); tab_view.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        self.create_display_tab(tab_view.add("Display"))
        self.create_actions_tab(tab_view.add("On-Add Actions"))
        button_frame = ctk.CTkFrame(self); button_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(button_frame, text="Help", command=self.show_help).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="right", padx=10)
        ctk.CTkButton(button_frame, text="Save", command=self.save, fg_color="green").pack(side="right", padx=10)
    def create_display_tab(self, tab):
        tab.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(tab, text="Indent Before Step:").grid(row=0, column=0, padx=10, pady=10, sticky="w")
        self.indent_before_entry = ctk.CTkEntry(tab, width=100); self.indent_before_entry.grid(row=0, column=1, padx=10, pady=10, sticky="w")
        self.indent_before_entry.insert(0, str(self.gui_data.get("indent_before", 0)))
        ctk.CTkLabel(tab, text="Indent After Step:").grid(row=1, column=0, padx=10, pady=10, sticky="w")
        self.indent_after_entry = ctk.CTkEntry(tab, width=100); self.indent_after_entry.grid(row=1, column=1, padx=10, pady=10, sticky="w")
        self.indent_after_entry.insert(0, str(self.gui_data.get("indent_after", 0)))
        self.hide_var = ctk.IntVar(value=1 if self.gui_data.get("hide_in_agent_list") else 0)
        ctk.CTkCheckBox(tab, text="Hide in Agent List (for partner agents)", variable=self.hide_var).grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="w")
    def create_actions_tab(self, tab):
        tab.grid_rowconfigure(1, weight=1); tab.grid_columnconfigure(0, weight=1)
        ctk.CTkButton(tab, text="+ Add 'Run Script' Action", command=self.add_script_action).pack(anchor="w", padx=10, pady=10)
        self.actions_frame = ctk.CTkScrollableFrame(tab, label_text="Scripts to run when this agent is added to a workflow");
        self.actions_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.refresh_actions_list()
    def refresh_actions_list(self):
        for widget_set in self.script_widgets: widget_set['frame'].destroy()
        self.script_widgets.clear()
        for action in self.gui_data.get('on_add', []):
            if action.get("action") == "run_script":
                script_name = action.get("script_name", "")
                script_code = "\n".join(self.gui_data.get('script_defs', {}).get(script_name, []))
                self.create_script_editor_widget(script_name, script_code)
    def add_script_action(self): self.create_script_editor_widget(f"new_script_{len(self.script_widgets) + 1}", "")
    def create_script_editor_widget(self, name, code):
        frame = ctk.CTkFrame(self.actions_frame); frame.pack(fill="x", expand=True, padx=5, pady=5)
        frame.grid_columnconfigure(1, weight=1)
        header = ctk.CTkFrame(frame); header.grid(row=0, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        header.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(header, text="Script Name:").grid(row=0, column=0, padx=(0,5))
        name_entry = ctk.CTkEntry(header); name_entry.insert(0, name); name_entry.grid(row=0, column=1, sticky="ew")
        remove_btn = ctk.CTkButton(header, text="Remove", fg_color="#D32F2F", hover_color="#B71C1C", width=80)
        remove_btn.grid(row=0, column=2, padx=(5,0))
        code_editor = CTkCodeEditor(frame, height=150); code_editor.insert("1.0", code)
        code_editor.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        widget_set = {'frame': frame, 'name': name_entry, 'code': code_editor, 'btn': remove_btn}
        remove_btn.configure(command=lambda w=widget_set: self.remove_script_action(w))
        self.script_widgets.append(widget_set)
    def remove_script_action(self, widget_set): widget_set['frame'].destroy(); self.script_widgets.remove(widget_set)
    def show_help(self): self.app_ref.show_help_modal("GUI Settings Help", """...""") # Content omitted for brevity
    def save(self):
        new_gui_data = {}
        try:
            if before_val := int(self.indent_before_entry.get() or 0): new_gui_data["indent_before"] = before_val
            if after_val := int(self.indent_after_entry.get() or 0): new_gui_data["indent_after"] = after_val
            if self.hide_var.get() == 1: new_gui_data["hide_in_agent_list"] = True
            new_on_add, new_script_defs = [], {}
            for widgets in self.script_widgets:
                if (name := widgets['name'].get().strip()) and (code := widgets['code'].get("1.0", "end-1c").strip()):
                    new_on_add.append({"action": "run_script", "script_name": name})
                    new_script_defs[name] = code.split('\n')
            if new_on_add: new_gui_data["on_add"] = new_on_add
            if new_script_defs: new_gui_data["script_defs"] = new_script_defs
            self.gui_data = new_gui_data or None
            self.saved = True; self.destroy()
        except ValueError: messagebox.showerror("Invalid Input", "Indent values must be integers.")
        except Exception as e: messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    def get_result(self): return self.gui_data
    def cancel(self): self.destroy()

if __name__ == "__main__":
    # --- MINIMAL CHANGES START HERE ---

    # 1. PARSE ARGUMENTS FIRST
    parser = argparse.ArgumentParser(description="A visual editor for the Dynamic Agent Workflow system.")
    parser.add_argument(
        '--config',
        default='config.json',
        help='Path to the configuration file to load. Default: config.json'
    )
    parser.add_argument(
        '--lib-path',
        help='Path to the directory containing dynamic_workflows_agents.py. If not provided, it is assumed to be in the current working directory.'
    )
    args = parser.parse_args()

    # 2. PREPARE LIBRARY PATH BEFORE IMPORT
    if args.lib_path:
        # Add the specified directory to the front of the Python path
        sys.path.insert(0, os.path.abspath(args.lib_path))

    # 3. USE THE ORIGINAL, ROBUST IMPORT BLOCK
    try:
        import dynamic_workflows_agents
        from dynamic_workflows_agents import exec_agent, create_temp_workflow, setup_depth_manager
    except ImportError:
        messagebox.showerror("Import Error", "Could not import the core workflow engine from 'dynamic_workflows_agents.py'. The 'Run' feature will be disabled.  You can specify a directory path to dynamic_workflows_agents.py using the --lib-path option on the command line.")
        dynamic_workflows_agents = None
        def exec_agent(**kwargs): return ({}, {"status":{"value":-1, "reason":"Core library not found"}})
        def setup_depth_manager(**kwargs): pass

    # --- APPLICATION STARTUP ---
    ctk.set_appearance_mode("System")
    ctk.set_default_color_theme("blue")
    
    # 4. PASS THE PARSED CONFIG PATH TO THE APP
    app = App(config_path=args.config)
    
    app.mainloop()

    # --- MINIMAL CHANGES END HERE ---
