# editor_app.py
import customtkinter as ctk
from tkinter import messagebox
import json
import copy
import time
import re
from config_manager import ConfigManager
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token

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

# --- API Class for GUI Scripts ---
class GuiApi:
    def __init__(self, workflow_editor_frame, source_step_index):
        self.editor = workflow_editor_frame; self.source_index = source_step_index
        self.app_ref = self.editor.app_ref; self.steps = self.editor.data.get("steps", [])
    def add_partner_agent(self, agent_name):
        if not agent_name: return
        agent_info = self.app_ref.config.get_agent_data(agent_name)
        if not agent_info: return
        new_params = {key: f"${key}" for key in agent_info.get("inputs", [])}
        new_step = {"agent": agent_name, "params": new_params, "output": agent_info.get("outputs", ["output"]).copy()}
        self.steps.insert(self.source_index + 1, new_step); print(f"GUI API: Added partner '{agent_name}'.")
    def find_partner_step(self, agent_name):
        for i in range(self.source_index + 1, len(self.steps)):
            if self.steps[i].get("agent") == agent_name: return self.steps[i], i
        return None, -1
    def get_step(self, index):
        if 0 <= index < len(self.steps): return self.steps[index]
        return None
    def set_step_outputs(self, index, new_outputs):
        step = self.get_step(index)
        if step: step['output'] = new_outputs
    def set_step_params(self, index, new_params):
        step = self.get_step(index)
        if step: step['params'] = new_params
    def get_agent_def(self, agent_name):
        return self.app_ref.config.get_agent_data(agent_name) or {}
    def generate_unique_id(self):
        return str(int(time.time()))[-6:]

# --- Editor for "workflow" agents ---
class WorkflowEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref):
        super().__init__(master, agent_name, agent_data, app_ref)
        tab_view = ctk.CTkTabview(self); tab_view.grid(row=0, column=0, sticky="nsew")
        self.create_settings_tab(tab_view.add("Settings"))
        self.create_inputs_tab(tab_view.add("Inputs"))
        self.create_optionals_tab(tab_view.add("Optional Inputs"))
        self.create_outputs_tab(tab_view.add("Outputs"))
        self.create_steps_tab(tab_view.add("Steps"))

    def create_settings_tab(self, tab):
        help_btn = self._create_help_button(tab, "Settings for this workflow agent.\n\n- Agent Name: The unique identifier for this agent.\n- Help Text: A description of what this agent does.\n- Return on Fail: If checked, the entire workflow will stop if any step inside it fails.")
        help_btn.place(relx=0.98, rely=0.02, anchor="ne")
        ctk.CTkLabel(tab, text="Agent Name:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.name_entry = ctk.CTkEntry(tab); self.name_entry.insert(0, self.agent_name); self.name_entry.pack(fill="x", padx=10, pady=5)
        ctk.CTkLabel(tab, text="Help Text:", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=10, pady=(10, 0))
        self.help_text = ctk.CTkTextbox(tab, height=200); self.help_text.insert("1.0", self.data.get("help", "")); self.help_text.pack(fill="x", padx=10, pady=5)
        fail_frame = ctk.CTkFrame(tab, fg_color="transparent"); fail_frame.pack(fill="x", padx=10, pady=10)
        self.fail_check_var = ctk.IntVar(value=self.data.get("return_on_fail", 0))
        self.fail_check = ctk.CTkCheckBox(fail_frame, text="Return on Fail", variable=self.fail_check_var); self.fail_check.pack(side="left")

    def create_inputs_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.inputs_frame = ListEditorFrame(tab, "Required Inputs", self.data.get("inputs", [])); self.inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_optionals_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.optionals_frame = ListEditorFrame(tab, "Optional Inputs", self.data.get("optional_inputs", [])); self.optionals_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_outputs_tab(self, tab):
        help_btn = self._create_help_button(tab, self._get_io_help_text()); help_btn.pack(anchor="ne", padx=5, pady=5)
        self.outputs_frame = ListEditorFrame(tab, "Outputs", self.data.get("outputs", [])); self.outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_steps_tab(self, tab):
        tab.grid_rowconfigure(1, weight=1); tab.grid_columnconfigure(0, weight=1)
        help_btn = self._create_help_button(tab, "This is the core of the workflow.\n\n- Click agents from the left list to add them as steps.\n- Use the Edit button to configure a step's parameters.\n- Use the arrows to reorder steps.\n- Use the 'X' to remove a step.")
        help_btn.place(relx=0.98, rely=0.02, anchor="ne")
        ctk.CTkLabel(tab, text="Workflow Steps").grid(row=0, column=0, pady=(5,0))
        self.steps_frame = ctk.CTkScrollableFrame(tab); self.steps_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.refresh_steps_list()

    def move_step(self, index, direction):
        steps = self.data.get('steps', [])
        if not (0 <= index < len(steps)): return
        new_index = index + direction
        if not (0 <= new_index < len(steps)): return
        steps.insert(new_index, steps.pop(index)); self.refresh_steps_list()
    def remove_step(self, index): self.data['steps'].pop(index); self.refresh_steps_list()
    def add_agent_as_step(self, agent_name):
        agent_info = self.app_ref.config.get_agent_data(agent_name)
        if not agent_info: return
        new_params = {key: f"${key}" for key in agent_info.get("inputs", [])}
        new_step = {"agent": agent_name, "params": new_params, "output": agent_info.get("outputs", ["output"]).copy()}
        self.data.setdefault("steps", []).append(new_step)
        added_step_index = len(self.data.get("steps", [])) - 1
        gui_conf = agent_info.get("gui", {})
        if gui_conf.get("on_add"): self._process_gui_directives(gui_conf, added_step_index)
        self.refresh_steps_list()
    def _process_gui_directives(self, gui_conf, source_step_index):
        api = GuiApi(self, source_step_index)
        script_defs = gui_conf.get("script_defs", {})
        for directive in gui_conf.get("on_add", []):
            if directive.get("action") == "run_script":
                script_name = directive.get("script_name")
                script_code_lines = script_defs.get(script_name)
                if not script_code_lines: print(f"GUI directive error: Script '{script_name}' not found."); continue
                script_code = "\n".join(script_code_lines)
                try: exec(script_code, {"api": api})
                except Exception as e: print(f"Error executing GUI script '{script_name}': {e}")

    # --- FIX START: Logic added to highlight invalid steps in red ---
    def refresh_steps_list(self):
        for widget in self.steps_frame.winfo_children(): widget.destroy()
        steps = self.data.get("steps", []); current_indent = 0; indent_char = "    "
        for i, step in enumerate(steps):
            agent_name = step.get('agent', 'Unknown')
            agent_def = self.app_ref.config.get_agent_data(agent_name)
            
            gui_hints = (agent_def or {}).get("gui", {})
            indent_modifier_before = gui_hints.get("indent_before", 0)
            current_indent = max(0, current_indent + indent_modifier_before)
            
            step_frame = ctk.CTkFrame(self.steps_frame); step_frame.pack(fill="x", pady=2)
            step_frame.grid_columnconfigure(3, weight=1)
            
            edit_btn = ctk.CTkButton(step_frame, text="Edit", width=60, command=lambda index=i: self.app_ref.open_step_editor(index)); edit_btn.grid(row=0, column=0, padx=5, pady=5)
            up_btn = ctk.CTkButton(step_frame, text="▲", width=30, command=lambda index=i: self.move_step(index, -1)); up_btn.grid(row=0, column=1, padx=(5,0), pady=5)
            down_btn = ctk.CTkButton(step_frame, text="▼", width=30, command=lambda index=i: self.move_step(index, 1)); down_btn.grid(row=0, column=2, padx=(1,5), pady=5)
            
            label_text = f"{indent_char * current_indent}{i}. {agent_name}"
            step_label = ctk.CTkLabel(step_frame, text=label_text)
            step_label.grid(row=0, column=3, padx=10, pady=5, sticky="w")
            
            # If the agent definition was not found, highlight the label in red.
            if not agent_def:
                step_label.configure(text_color="red", text=f"{label_text} (not found)")
            
            remove_btn = ctk.CTkButton(step_frame, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda index=i: self.remove_step(index)); remove_btn.grid(row=0, column=4, padx=5, pady=5)
            
            indent_modifier_after = gui_hints.get("indent_after", 0)
            current_indent = max(0, current_indent + indent_modifier_after)
    # --- FIX END ---
            
    def get_data(self):
        updated_data = copy.deepcopy(self.data)
        updated_data['name'] = self.name_entry.get().strip()
        updated_data['help'] = self.help_text.get("1.0", "end-1c").strip()
        updated_data['inputs'] = self.inputs_frame.get_data()
        updated_data['optional_inputs'] = self.optionals_frame.get_data()
        updated_data['outputs'] = self.outputs_frame.get_data()
        updated_data['return_on_fail'] = self.fail_check_var.get()
        return updated_data

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
    def __init__(self):
        super().__init__(); self.title("Agent Workflow Editor"); self.geometry("1400x800")
        self.config = ConfigManager(); self.current_agent_name = None; self.editor_frame_instance = None
        self.search_text = ctk.StringVar(); self.search_text.trace("w", self.on_search_changed)
        self.show_hidden_agents_var = ctk.IntVar(value=0)
        self.grid_columnconfigure(0, weight=0); self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        self.create_agent_list_panel(); self.create_editor_panel(); self.create_modal_overlay(); self.refresh_agent_list(); self.show_welcome_message()
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
        self.action_bar = ctk.CTkFrame(self, fg_color="transparent"); self.action_bar.grid(row=1, column=1, padx=10, pady=10, sticky="sew")
        self.cancel_btn = ctk.CTkButton(self.action_bar, text="Cancel", command=self.show_welcome_message)
        self.save_btn = ctk.CTkButton(self.action_bar, text="Save Agent", command=self.save_agent, fg_color="green")
    def create_modal_overlay(self):
        self.overlay = ctk.CTkFrame(self, fg_color=("#000000", "#000000")); self.overlay.lower()
        self.overlay_label = ctk.CTkLabel(self.overlay, text="Editing Step...\nMain window is locked.", font=ctk.CTkFont(size=24, weight="bold"))
    def show_overlay(self): self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1); self.overlay_label.place(relx=0.5, rely=0.5, anchor="center"); self.overlay.lift()
    def hide_overlay(self): self.overlay.place_forget()
    def on_search_changed(self, *args): self.refresh_agent_list()
    def refresh_agent_list(self):
        for widget in self.agent_scroll_frame.winfo_children(): widget.destroy()
        search_term = self.search_text.get().lower()
        show_hidden = self.show_hidden_agents_var.get() == 1
        for name in self.config.get_agent_names():
            agent_data = self.config.get_agent_data(name)
            if agent_data.get("gui", {}).get("hide_in_agent_list", False) and not show_hidden: continue
            if search_term and search_term not in name.lower(): continue
            agent_type = agent_data.get("type", "json")
            prefix = {"workflow": "W", "proc": "P", "template": "T"}.get(agent_type, "J")
            row = ctk.CTkFrame(self.agent_scroll_frame, fg_color="transparent"); row.pack(fill="x", padx=2, pady=2)
            del_btn = ctk.CTkButton(row, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda n=name: self.delete_agent(agent_name=n, confirm=True)); del_btn.pack(side="right")
            btn = ctk.CTkButton(row, text=f"[{prefix}] {name}", command=lambda n=name: self.on_agent_list_click(n), anchor="w"); btn.pack(side="left", fill="x", expand=True)
    def on_agent_list_click(self, agent_name):
        if self.editor_frame_instance and isinstance(self.editor_frame_instance, WorkflowEditorFrame):
            self.editor_frame_instance.add_agent_as_step(agent_name)
        else: self.select_agent(agent_name)
    def add_new_agent(self, choice):
        agent_type = choice.lower(); new_name = self.config.create_new_agent(agent_type)
        self.config.save(); self.refresh_agent_list(); self.select_agent(new_name)
    def select_agent(self, agent_name): self.current_agent_name = agent_name; self.build_editor_form()
    def show_welcome_message(self):
        if self.editor_frame_instance: self.editor_frame_instance.destroy()
        self.current_agent_name = None; self.editor_frame_instance = None
        label = ctk.CTkLabel(self.editor_container, text="Select an agent to edit or create a new one.", font=ctk.CTkFont(size=24)); label.place(relx=0.5, rely=0.5, anchor="center")
        self.action_bar.grid_remove()
    def build_editor_form(self):
        if self.editor_frame_instance: self.editor_frame_instance.destroy()
        self.action_bar.grid(); self.save_btn.pack(side="right", padx=10, pady=10); self.cancel_btn.pack(side="right", padx=0, pady=10)
        agent_data = self.config.get_agent_data(self.current_agent_name)
        editor_class = {"workflow": WorkflowEditorFrame, "proc": ProcEditorFrame, "template": TemplateEditorFrame}.get(agent_data.get("type"), JsonEditorFrame)
        self.editor_frame_instance = editor_class(self.editor_container, self.current_agent_name, copy.deepcopy(agent_data), self)
        self.editor_frame_instance.grid(row=0, column=0, sticky="nsew")
    def save_agent(self):
        if not self.editor_frame_instance or not self.current_agent_name: return
        updated_data = self.editor_frame_instance.get_data();
        if updated_data is None: return
        
        new_name = updated_data.pop('name', self.current_agent_name)
        if not new_name: messagebox.showerror("Error", "Agent name cannot be empty."); return
        if new_name != self.current_agent_name:
            if not self.config.rename_agent(self.current_agent_name, new_name): messagebox.showerror("Error", f"Agent name '{new_name}' already exists."); return
            self.current_agent_name = new_name
        self.config.update_agent(self.current_agent_name, updated_data); self.config.save(); self.show_toast(f"Agent '{self.current_agent_name}' saved.")
        self.refresh_agent_list(); self.show_welcome_message()
    def delete_agent(self, agent_name=None, confirm=False):
        name_to_delete = agent_name;
        if not name_to_delete: return
        deps = self.config.check_agent_usage(name_to_delete)
        if deps: messagebox.showerror("Cannot Delete", f"'{name_to_delete}' is used by:\n- " + "\n- ".join(deps)); return
        if confirm and not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{name_to_delete}'? This cannot be undone."): return
        self.config.delete_agent(name_to_delete); self.config.save(); self.refresh_agent_list()
        if name_to_delete == self.current_agent_name: self.show_welcome_message()
    def open_global_config(self):
        modal = GlobalConfigEditorModal(self, self.config); self.wait_window(modal)
        if modal.saved: self.config.save(); self.show_toast("Global configuration saved successfully!")
    
    # --- FIX START: Logic added to show a popup on invalid agent edit attempts ---
    def open_step_editor(self, index):
        if not isinstance(self.editor_frame_instance, WorkflowEditorFrame): return

        parent_workflow_data = self.editor_frame_instance.get_data()
        if parent_workflow_data is None:
            messagebox.showerror("Error", "Could not read current workflow data. Please check for errors.")
            return

        step_data = parent_workflow_data["steps"][index]
        agent_name = step_data.get("agent", "")
        agent_def = self.config.get_agent_data(agent_name)

        # Check if the agent exists BEFORE opening the modal.
        if not agent_def:
            messagebox.showerror(
                "Agent Not Found",
                f"The agent '{agent_name}' used in step {index} could not be found.\n\n"
                "It may have been renamed or deleted. Please correct the agent name in the workflow."
            )
            return  # Stop execution here

        self.show_overlay()
        modal = StepEditorModal(self, index, step_data, agent_def, parent_workflow_data)
        self.wait_window(modal)
        self.hide_overlay()
        
        if modal.saved:
            self.editor_frame_instance.data["steps"][index] = modal.get_result()
            self.editor_frame_instance.refresh_steps_list()
    # --- FIX END ---

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

# --- Step Editor Modal Window ---
class StepEditorModal(ctk.CTkToplevel):
    # --- FIX START: Reverted signature to accept pre-validated agent_def ---
    def __init__(self, parent, index, step_data, agent_def, parent_workflow_data=None):
        super().__init__(parent)
        self.title(f"Edit Step {index}: {step_data.get('agent')}")
        self.geometry("1100x700")

        self.agent_def = agent_def or {}
    # --- FIX END ---

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
            for output_var in step.get('output', []):
                self.available_vars.add(output_var)

    def validate_entry_variables(self, entry_widget):
        text = entry_widget.get()
        found_vars = re.findall(r'\$(\w+)', text)
        
        if not text:
            entry_widget.configure(border_color=self.default_border_color, border_width=1)
            return
        if not found_vars:
            entry_widget.configure(border_color=self.literal_border_color, border_width=1)
            return
            
        is_valid = all(var in self.available_vars for var in found_vars)
        
        if not is_valid:
            entry_widget.configure(border_color="red", border_width=1)
        else:
            entry_widget.configure(border_color=self.default_border_color, border_width=1)

    def set_active_entry(self, entry_widget):
        self.active_entry = entry_widget

    def append_variable_to_active_entry(self, var_name):
        if self.active_entry:
            cursor_pos = self.active_entry.index(ctk.INSERT)
            variable_string = f"${var_name}"
            if cursor_pos > 0 and self.active_entry.get() and self.active_entry.get()[cursor_pos-1] not in (' ', '('):
                variable_string = " " + variable_string
            self.active_entry.insert(cursor_pos, variable_string)
            self.active_entry.focus()
            self.validate_entry_variables(self.active_entry)

    def populate_variable_selector(self):
        def create_var_button(parent, var_name):
            btn = ctk.CTkButton(parent, text=var_name, anchor="w", fg_color="gray",
                                command=lambda v=var_name: self.append_variable_to_active_entry(v))
            btn.pack(fill="x", padx=5, pady=2)

        for var in sorted(list(self.parent_workflow_data.get('inputs', []))):
            create_var_button(self.wf_req_inputs_frame, var)
        for var in sorted(list(self.parent_workflow_data.get('optional_inputs', []))):
            create_var_button(self.wf_opt_inputs_frame, var)

        for i in range(self.step_index):
            step = self.parent_workflow_data['steps'][i]
            agent_name = step.get('agent', 'Unknown')
            ctk.CTkLabel(self.step_outputs_frame, text=f"Step {i}: {agent_name}",
                         font=ctk.CTkFont(weight="bold")).pack(anchor="w", pady=(8, 2), padx=5)
            for output_var in step.get('output', []):
                create_var_button(self.step_outputs_frame, output_var)

    def create_widgets(self):
        selector_frame = ctk.CTkFrame(self, width=300)
        selector_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        selector_frame.grid_rowconfigure(0, weight=1); selector_frame.grid_columnconfigure(0, weight=1)
        selector_tabs = ctk.CTkTabview(selector_frame)
        selector_tabs.grid(row=0, column=0, sticky="nsew")

        workflow_inputs_tab = selector_tabs.add("Workflow Inputs")
        step_outputs_tab = selector_tabs.add("Step Outputs")

        self.wf_req_inputs_frame = ctk.CTkScrollableFrame(workflow_inputs_tab, label_text="Required")
        self.wf_req_inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.wf_opt_inputs_frame = ctk.CTkScrollableFrame(workflow_inputs_tab, label_text="Optional")
        self.wf_opt_inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.step_outputs_frame = ctk.CTkScrollableFrame(step_outputs_tab, label_text="Available Outputs from Previous Steps")
        self.step_outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

        form = ctk.CTkFrame(self); form.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        form.grid_rowconfigure(0, weight=1); form.grid_columnconfigure(0, weight=1)
        self.params_frame = ctk.CTkScrollableFrame(form, label_text="Parameters & Outputs"); self.params_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        
        button_frame = ctk.CTkFrame(self); button_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save", command=self.save, fg_color="green").pack(side="right", padx=10)
        
        self.populate_variable_selector()
        self.refresh_params_form()

    def add_optional_param(self, param_name: str):
        if param_name and "Add Optional" not in param_name:
            self.editing_data.setdefault("params", {})[param_name] = ""
            self.refresh_params_form()
            
    def refresh_params_form(self):
        for widget in self.params_frame.winfo_children(): widget.destroy()
        self.param_entries.clear()

        required_inputs = set(self.agent_def.get("inputs", []))
        optional_inputs = set(self.agent_def.get("optional_inputs", []))
        
        self.editing_data.setdefault('params', {})
        self.editing_data.setdefault('output', [])
        
        ctk.CTkLabel(self.params_frame, text="Outputs", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        for i, item in enumerate(self.editing_data['output']):
            entry = ctk.CTkEntry(self.params_frame); entry.insert(0, item); entry.pack(fill="x", padx=5, pady=2)
            self.param_entries[f'output_{i}'] = entry

        if required_inputs:
            ctk.CTkLabel(self.params_frame, text="Required Parameters", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(15, 0))
        for key in sorted(list(required_inputs)):
            if key not in self.editing_data["params"]:
                self.editing_data["params"][key] = f"${key}"
            
            value = self.editing_data["params"].get(key, "")
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent"); param_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(param_frame, text=key, width=200).pack(side="left", padx=5)
            
            value_entry = ctk.CTkEntry(param_frame); value_entry.insert(0, value)
            value_entry.pack(side="left", padx=5, expand=True, fill="x")
            value_entry.bind("<FocusIn>", lambda event, entry=value_entry: self.set_active_entry(entry))
            value_entry.bind("<KeyRelease>", lambda event, entry=value_entry: self.validate_entry_variables(entry))
            self.param_entries[f'param_val_{key}'] = value_entry
            self.validate_entry_variables(value_entry)

        current_param_keys = set(self.editing_data["params"].keys())
        optional_and_custom_keys = current_param_keys - required_inputs

        if optional_and_custom_keys:
            ctk.CTkLabel(self.params_frame, text="Optional / Custom Parameters", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(15, 0))

        for key in sorted(list(optional_and_custom_keys)):
            value = self.editing_data["params"].get(key, "")
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent"); param_frame.pack(fill="x", pady=2)
            
            if key in optional_inputs:
                ctk.CTkLabel(param_frame, text=key, width=200).pack(side="left", padx=5)
            else:
                key_entry = ctk.CTkEntry(param_frame, width=200); key_entry.insert(0, key); key_entry.pack(side="left", padx=5)
                self.param_entries[f'param_key_{key}'] = key_entry

            value_entry = ctk.CTkEntry(param_frame); value_entry.insert(0, value);
            value_entry.pack(side="left", padx=5, expand=True, fill="x")
            value_entry.bind("<FocusIn>", lambda event, entry=value_entry: self.set_active_entry(entry))
            value_entry.bind("<KeyRelease>", lambda event, entry=value_entry: self.validate_entry_variables(entry))
            self.param_entries[f'param_val_{key}'] = value_entry
            self.validate_entry_variables(value_entry)

            remove_btn = ctk.CTkButton(param_frame, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda k=key: self.remove_param(k))
            remove_btn.pack(side="left", padx=5)
            
        available_options = sorted(list(optional_inputs - current_param_keys))
        if available_options:
            placeholder = "Add Optional Input..."
            option_menu = ctk.CTkOptionMenu(self.params_frame, values=[placeholder] + available_options, command=self.add_optional_param)
            option_menu.set(placeholder)
            option_menu.pack(pady=10, padx=5, anchor="w")

    def remove_param(self, key):
        required_inputs = set(self.agent_def.get("inputs", []))
        if key in self.editing_data.get("params", {}) and key not in required_inputs:
            del self.editing_data["params"][key]
        self.refresh_params_form()
        
    def get_form_data(self):
        new_outputs = []; new_params = {}
        original_output_len = len(self.editing_data.get('output',[]))
        for i in range(original_output_len): 
            new_outputs.append(self.param_entries[f'output_{i}'].get())
        
        all_rendered_keys = {k.replace('param_val_', '') for k in self.param_entries if k.startswith('param_val_')}
        
        for key_ref in all_rendered_keys:
            new_val = self.param_entries[f'param_val_{key_ref}'].get()
            
            if f'param_key_{key_ref}' in self.param_entries:
                new_key = self.param_entries[f'param_key_{key_ref}'].get()
            else:
                new_key = key_ref
            
            if new_key:
                new_params[new_key] = new_val

        self.editing_data['output'] = new_outputs
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
        on_add = self.gui_data.get('on_add', [])
        script_defs = self.gui_data.get('script_defs', {})
        for action in on_add:
            if action.get("action") == "run_script":
                script_name = action.get("script_name", "")
                script_code = "\n".join(script_defs.get(script_name, []))
                self.create_script_editor_widget(script_name, script_code)
    def add_script_action(self):
        num = len(self.script_widgets) + 1; self.create_script_editor_widget(f"new_script_{num}", "")
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
    def remove_script_action(self, widget_set):
        widget_set['frame'].destroy(); self.script_widgets.remove(widget_set)
    def show_help(self):
        help_text = """
The 'gui' tag controls editor behavior for an agent.

<Display Tab>
- Indent Before/After: Controls visual indentation in the workflow step list. Use 1 and -1 for starting and ending a block.
- Hide in Agent List: If checked, this agent won't appear in the main list, useful for partner agents like 'loop_end'.

<On-Add Actions Tab>
This powerful feature lets an agent run its own setup script when added to a workflow. The script is defined here and triggered by a 'run_script' action in the config.

Available `api` functions for your script:
-------------------------------------------------
- api.add_partner_agent('agent_name')
- api.find_partner_step('agent_name') -> (step_dict, index)
- api.get_step(index) -> step_dict
- api.set_step_outputs(index, ['out1', 'out2'])
- api.set_step_params(index, {'param1': '$val1'})
- api.get_agent_def('agent_name') -> agent_def_dict
- api.generate_unique_id() -> '123456'
"""
        help_window = ctk.CTkToplevel(self)
        help_window.title("GUI Settings Help"); help_window.geometry("700x500")
        help_window.transient(self); help_window.grab_set()
        textbox = ctk.CTkTextbox(help_window, wrap="word", font=("", 14))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)
        textbox.insert("1.0", help_text); textbox.configure(state="disabled")
    def save(self):
        new_gui_data = {}
        try:
            before_val = int(self.indent_before_entry.get() or 0)
            after_val = int(self.indent_after_entry.get() or 0)
            if before_val != 0: new_gui_data["indent_before"] = before_val
            if after_val != 0: new_gui_data["indent_after"] = after_val
            if self.hide_var.get() == 1: new_gui_data["hide_in_agent_list"] = True
            new_on_add, new_script_defs = [], {}
            for widgets in self.script_widgets:
                name = widgets['name'].get().strip()
                code = widgets['code'].get("1.0", "end-1c").strip()
                if name and code:
                    new_on_add.append({"action": "run_script", "script_name": name})
                    new_script_defs[name] = code.split('\n')
            if new_on_add: new_gui_data["on_add"] = new_on_add
            if new_script_defs: new_gui_data["script_defs"] = new_script_defs
            self.gui_data = new_gui_data if new_gui_data else None
            self.saved = True; self.destroy()
        except ValueError: messagebox.showerror("Invalid Input", "Indent values must be integers.")
        except Exception as e: messagebox.showerror("Error", f"An unexpected error occurred: {e}")
    def get_result(self): return self.gui_data
    def cancel(self): self.destroy()

if __name__ == "__main__":
    ctk.set_appearance_mode("System"); ctk.set_default_color_theme("blue"); app = App(); app.mainloop()