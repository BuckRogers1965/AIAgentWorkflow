# editor_app.py
import customtkinter as ctk
from tkinter import messagebox
import json
import copy
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
        self.textbox._textbox.configure(yscrollcommand=self.sync_scroll); self.line_numbers._textbox.configure(yscrollcommand=self.sync_scroll)
    
    def sync_scroll(self, *args): self.textbox._textbox.yview_moveto(args[0]); self.line_numbers._textbox.yview_moveto(args[0]); return "break"
    def on_return(self, event): self.textbox.insert(ctk.INSERT, "\n"); self.update_syntax_highlighting(); return "break"
    def on_key_release(self, event=None): self.update_syntax_highlighting()
    def update_line_numbers(self):
        self.line_numbers.configure(state="normal"); self.line_numbers.delete("1.0", "end")
        line_count = int(self.textbox.index("end-1c").split('.')[0]); line_numbers_string = "\n".join(str(i) for i in range(1, line_count + 1))
        self.line_numbers.insert("1.0", line_numbers_string); self.line_numbers.configure(state="disabled")
    def update_syntax_highlighting(self, event=None):
        for tag in self.tag_colors.keys(): self.textbox.tag_remove(str(tag), "1.0", "end")
        text = self.textbox.get("1.0", "end-1c")
        if not text: self.update_line_numbers(); return
        start_pos = "1.0"
        for token, content in lex(text, self.language_lexer):
            end_pos = f"{start_pos}+{len(content)}c"
            base_token = token
            while base_token not in self.tag_colors and base_token.parent: base_token = base_token.parent
            if base_token in self.tag_colors: self.textbox.tag_add(str(base_token), start_pos, end_pos)
            start_pos = end_pos
        self.update_line_numbers()
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
class BaseEditorFrame(ctk.CTkScrollableFrame):
    def __init__(self, master, agent_data, app_ref):
        super().__init__(master); self.data = agent_data; self.app_ref = app_ref
    def get_data(self): raise NotImplementedError

# --- Editor for "workflow" agents ---
class WorkflowEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_data, app_ref):
        super().__init__(master, agent_data, app_ref)
        ctk.CTkLabel(self, text="Help Text").pack(anchor="w", padx=5)
        self.help_text = ctk.CTkTextbox(self, height=80); self.help_text.insert("1.0", self.data.get("help", "")); self.help_text.pack(fill="x", expand=True, padx=5, pady=(0, 10))
        self.inputs_frame = ListEditorFrame(self, "Inputs", self.data.get("inputs", [])); self.inputs_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self.optionals_frame = ListEditorFrame(self, "Optional Inputs", self.data.get("optional_inputs", [])); self.optionals_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self.outputs_frame = ListEditorFrame(self, "Outputs", self.data.get("outputs", [])); self.outputs_frame.pack(fill="x", expand=True, padx=5, pady=5)
        fail_frame = ctk.CTkFrame(self, fg_color="transparent"); fail_frame.pack(fill="x", padx=5, pady=10)
        self.fail_check_var = ctk.IntVar(value=self.data.get("return_on_fail", 0))
        self.fail_check = ctk.CTkCheckBox(fail_frame, text="Return on Fail", variable=self.fail_check_var); self.fail_check.pack(side="left")
        ctk.CTkLabel(self, text="Workflow Steps", font=ctk.CTkFont(size=16, weight="bold")).pack(anchor="w", pady=(20, 5), padx=5)
        self.steps_frame = ctk.CTkFrame(self); self.steps_frame.pack(fill="x", expand=True, padx=5)
        self.refresh_steps_list()
    def move_step(self, index, direction):
        steps = self.data.get('steps', []);
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
        self.data.setdefault("steps", []).append(new_step); self.refresh_steps_list()
    def refresh_steps_list(self):
        for widget in self.steps_frame.winfo_children(): widget.destroy()
        steps = self.data.get("steps", [])
        for i, step in enumerate(steps):
            step_frame = ctk.CTkFrame(self.steps_frame); step_frame.pack(fill="x", pady=2)
            step_frame.grid_columnconfigure(3, weight=1)
            edit_btn = ctk.CTkButton(step_frame, text="Edit", width=60, command=lambda index=i: self.app_ref.open_step_editor(index)); edit_btn.grid(row=0, column=0, padx=5, pady=5)
            up_btn = ctk.CTkButton(step_frame, text="▲", width=30, command=lambda index=i: self.move_step(index, -1)); up_btn.grid(row=0, column=1, padx=(5,0), pady=5)
            down_btn = ctk.CTkButton(step_frame, text="▼", width=30, command=lambda index=i: self.move_step(index, 1)); down_btn.grid(row=0, column=2, padx=(1,5), pady=5)
            ctk.CTkLabel(step_frame, text=f"{i}. {step.get('agent', 'Unknown')}").grid(row=0, column=3, padx=10, pady=5, sticky="w")
            remove_btn = ctk.CTkButton(step_frame, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda index=i: self.remove_step(index)); remove_btn.grid(row=0, column=4, padx=5, pady=5)
    def get_data(self):
        self.data['help'] = self.help_text.get("1.0", "end-1c").strip()
        self.data['inputs'] = self.inputs_frame.get_data(); self.data['optional_inputs'] = self.optionals_frame.get_data(); self.data['outputs'] = self.outputs_frame.get_data()
        self.data['return_on_fail'] = self.fail_check_var.get(); return self.data

# --- Editors for "proc" and "template" ---
class ProcEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_data, app_ref):
        super().__init__(master, agent_data, app_ref)
        ctk.CTkLabel(self, text="Help Text").pack(anchor="w", padx=5)
        self.help_text = ctk.CTkTextbox(self, height=80); self.help_text.insert("1.0", self.data.get("help", "")); self.help_text.pack(fill="x", expand=True, padx=5, pady=(0, 10))
        self.inputs_frame = ListEditorFrame(self, "Inputs", self.data.get("inputs", [])); self.inputs_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self.optionals_frame = ListEditorFrame(self, "Optional Inputs", self.data.get("optional_inputs", [])); self.optionals_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self.outputs_frame = ListEditorFrame(self, "Outputs", self.data.get("outputs", [])); self.outputs_frame.pack(fill="x", expand=True, padx=5, pady=5)
        ctk.CTkLabel(self, text="Function Name").pack(anchor="w", padx=5, pady=(10,0))
        self.func_name_entry = ctk.CTkEntry(self); self.func_name_entry.insert(0, self.data.get("function", "")); self.func_name_entry.pack(fill="x", padx=5, pady=5)
        ctk.CTkLabel(self, text="Function Definition").pack(anchor="w", padx=5, pady=(10,0))
        self.func_def_text = CTkCodeEditor(self, height=300); self.func_def_text.insert("1.0", self.data.get("function_def", "")); self.func_def_text.pack(fill="both", expand=True, padx=5, pady=5)
    def get_data(self):
        self.data['help'] = self.help_text.get("1.0", "end-1c").strip()
        self.data['inputs'] = self.inputs_frame.get_data(); self.data['optional_inputs'] = self.optionals_frame.get_data(); self.data['outputs'] = self.outputs_frame.get_data()
        self.data['function'] = self.func_name_entry.get(); self.data['function_def'] = self.func_def_text.get("1.0", "end-1c").strip(); return self.data
class TemplateEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_data, app_ref):
        super().__init__(master, agent_data, app_ref)
        ctk.CTkLabel(self, text="Help Text").pack(anchor="w", padx=5)
        self.help_text = ctk.CTkTextbox(self, height=80); self.help_text.insert("1.0", self.data.get("help", "")); self.help_text.pack(fill="x", expand=True, padx=5, pady=(0, 10))
        self.inputs_frame = ListEditorFrame(self, "Inputs", self.data.get("inputs", [])); self.inputs_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self.optionals_frame = ListEditorFrame(self, "Optional Inputs", self.data.get("optional_inputs", [])); self.optionals_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self.outputs_frame = ListEditorFrame(self, "Outputs", self.data.get("outputs", [])); self.outputs_frame.pack(fill="x", expand=True, padx=5, pady=5)
        ctk.CTkLabel(self, text="Prompt").pack(anchor="w", padx=5, pady=(10,0))
        self.prompt_text = ctk.CTkTextbox(self, height=200); self.prompt_text.insert("1.0", self.data.get("prompt", "")); self.prompt_text.pack(fill="both", expand=True, padx=5, pady=5)
    def get_data(self):
        self.data['help'] = self.help_text.get("1.0", "end-1c").strip()
        self.data['inputs'] = self.inputs_frame.get_data(); self.data['optional_inputs'] = self.optionals_frame.get_data(); self.data['outputs'] = self.outputs_frame.get_data()
        self.data['prompt'] = self.prompt_text.get("1.0", "end-1c").strip(); return self.data
class JsonEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_data, app_ref):
        super().__init__(master, agent_data, app_ref)
        self.textbox = ctk.CTkTextbox(self, font=("monospace", 12)); self.textbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.textbox.insert("1.0", json.dumps(self.data, indent=2))
    def get_data(self):
        try: return json.loads(self.textbox.get("1.0", "end-1c"))
        except json.JSONDecodeError as e: messagebox.showerror("JSON Error", f"Invalid JSON: {e}"); return None

# --- Main Application Window ---
class App(ctk.CTk):
    def __init__(self):
        super().__init__(); self.title("Agent Workflow Editor"); self.geometry("1400x800")
        self.config = ConfigManager(); self.current_agent_name = None; self.editor_frame_instance = None
        self.grid_columnconfigure(0, weight=0); self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.create_agent_list_panel(); self.create_editor_panel(); self.create_modal_overlay(); self.refresh_agent_list(); self.show_welcome_message()

    def create_agent_list_panel(self):
        self.agent_list_frame = ctk.CTkFrame(self, width=320); self.agent_list_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="ns")
        self.agent_list_frame.grid_propagate(False)
        self.agent_list_frame.grid_rowconfigure(2, weight=1); self.agent_list_frame.grid_columnconfigure(0, weight=1)
        ctk.CTkLabel(self.agent_list_frame, text="Agents", font=ctk.CTkFont(size=20, weight="bold")).grid(row=0, column=0, padx=20, pady=(20, 10), sticky="w")
        add_menu = ctk.CTkOptionMenu(self.agent_list_frame, values=["Workflow", "Proc", "Template"], command=self.add_new_agent); add_menu.grid(row=1, column=0, padx=20, pady=10, sticky="ew"); add_menu.set("Create New Agent...")
        self.agent_scroll_frame = ctk.CTkScrollableFrame(self.agent_list_frame, label_text="Existing Agents"); self.agent_scroll_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")

    def create_editor_panel(self):
        self.editor_container = ctk.CTkFrame(self); self.editor_container.grid(row=0, column=1, padx=10, pady=(10,0), sticky="nsew")
        self.editor_container.grid_rowconfigure(1, weight=1); self.editor_container.grid_columnconfigure(0, weight=1)
        self.action_bar = ctk.CTkFrame(self, fg_color="transparent"); self.action_bar.grid(row=1, column=1, padx=10, pady=10, sticky="sew")
        self.cancel_btn = ctk.CTkButton(self.action_bar, text="Cancel", command=self.show_welcome_message)
        self.save_btn = ctk.CTkButton(self.action_bar, text="Save Agent", command=self.save_agent, fg_color="green")

    def create_modal_overlay(self):
        self.overlay = ctk.CTkFrame(self, fg_color=("#000000", "#000000"))
        self.overlay.lower()
        self.overlay_label = ctk.CTkLabel(self.overlay, text="Editing Step...\nMain window is locked.", font=ctk.CTkFont(size=24, weight="bold"))

    def show_overlay(self):
        self.overlay.place(relx=0, rely=0, relwidth=1, relheight=1)
        self.overlay_label.place(relx=0.5, rely=0.5, anchor="center")
        self.overlay.lift()

    def hide_overlay(self):
        self.overlay.place_forget()

    def refresh_agent_list(self):
        for widget in self.agent_scroll_frame.winfo_children(): widget.destroy()
        for name in self.config.get_agent_names():
            agent_type = self.config.get_agent_data(name).get("type", "json"); prefix = {"workflow": "W", "proc": "P", "template": "T"}.get(agent_type, "J")
            row = ctk.CTkFrame(self.agent_scroll_frame, fg_color="transparent"); row.pack(fill="x", padx=2, pady=2)
            del_btn = ctk.CTkButton(row, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda n=name: self.delete_agent(agent_name=n, confirm=True)); del_btn.pack(side="right")
            btn = ctk.CTkButton(row, text=f"[{prefix}] {name}", command=lambda n=name: self.on_agent_list_click(n), anchor="w"); btn.pack(side="left", fill="x", expand=True)
            
    def on_agent_list_click(self, agent_name):
        if self.editor_frame_instance and isinstance(self.editor_frame_instance, WorkflowEditorFrame):
            self.editor_frame_instance.add_agent_as_step(agent_name)
        else:
            self.select_agent(agent_name)

    def add_new_agent(self, choice):
        agent_type = choice.lower(); new_name = self.config.create_new_agent(agent_type)
        self.config.save(); self.refresh_agent_list(); self.select_agent(new_name)

    def select_agent(self, agent_name): self.current_agent_name = agent_name; self.build_editor_form()
        
    def show_welcome_message(self):
        for widget in self.editor_container.winfo_children(): widget.destroy()
        self.current_agent_name = None; self.editor_frame_instance = None
        label = ctk.CTkLabel(self.editor_container, text="Select an agent to edit or create a new one.", font=ctk.CTkFont(size=24)); label.place(relx=0.5, rely=0.5, anchor="center")
        self.action_bar.grid_remove()

    def build_editor_form(self):
        for widget in self.editor_container.winfo_children(): widget.destroy()
        self.action_bar.grid(); self.save_btn.pack(side="right", padx=10, pady=10); self.cancel_btn.pack(side="right", padx=0, pady=10)
        agent_data = self.config.get_agent_data(self.current_agent_name)
        name_frame = ctk.CTkFrame(self.editor_container); name_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkLabel(name_frame, text="Agent Name:", font=ctk.CTkFont(size=16, weight="bold")).pack(side="left", padx=10)
        self.name_entry = ctk.CTkEntry(name_frame, font=ctk.CTkFont(size=16)); self.name_entry.insert(0, self.current_agent_name)
        self.name_entry.pack(side="left", fill="x", expand=True, padx=10, pady=10)
        editor_class = {"workflow": WorkflowEditorFrame, "proc": ProcEditorFrame, "template": TemplateEditorFrame}.get(agent_data.get("type"), JsonEditorFrame)
        self.editor_frame_instance = editor_class(self.editor_container, agent_data, self); self.editor_frame_instance.pack(fill="both", expand=True, padx=10, pady=(0,10))

    def save_agent(self):
        if not self.editor_frame_instance or not self.current_agent_name: return
        updated_data = self.editor_frame_instance.get_data()
        if updated_data is None: return
        new_name = self.name_entry.get().strip()
        if not new_name: messagebox.showerror("Error", "Agent name cannot be empty."); return
        if new_name != self.current_agent_name:
            if not self.config.rename_agent(self.current_agent_name, new_name): messagebox.showerror("Error", f"Agent name '{new_name}' already exists."); return
            self.current_agent_name = new_name
        self.config.update_agent(self.current_agent_name, updated_data)
        self.config.save(); self.show_toast(f"Agent '{self.current_agent_name}' saved.")
        self.refresh_agent_list(); self.show_welcome_message()

    def delete_agent(self, agent_name=None, confirm=False):
        name_to_delete = agent_name
        if not name_to_delete: return
        deps = self.config.check_agent_usage(name_to_delete)
        if deps: messagebox.showerror("Cannot Delete", f"'{name_to_delete}' is used by:\n- " + "\n- ".join(deps)); return
        if confirm and not messagebox.askyesno("Confirm Delete", f"Are you sure you want to delete '{name_to_delete}'? This cannot be undone."): return
        self.config.delete_agent(name_to_delete); self.config.save(); self.refresh_agent_list()
        if name_to_delete == self.current_agent_name: self.show_welcome_message()

    def open_step_editor(self, index):
        if not isinstance(self.editor_frame_instance, WorkflowEditorFrame): return
        step_data = self.editor_frame_instance.data["steps"][index]
        agent_def = self.config.get_agent_data(step_data.get("agent", ""))
        
        self.show_overlay()
        modal = StepEditorModal(self, index, step_data, agent_def)
        self.wait_window(modal)
        self.hide_overlay()
        
        if modal.saved:
            self.editor_frame_instance.data["steps"][index] = modal.get_result()
            self.editor_frame_instance.refresh_steps_list()
            
    def show_toast(self, message):
        toast = ctk.CTkLabel(self, text=message, fg_color=("#333", "#555"), text_color="white", corner_radius=10, font=("", 14))
        toast.place(relx=0.5, rely=0.95, anchor="center")
        toast.lift()
        self.after(2500, toast.destroy)

# --- Step Editor Modal Window ---
class StepEditorModal(ctk.CTkToplevel):
    def __init__(self, parent, index, step_data, agent_def):
        super().__init__(parent); self.title(f"Edit Step {index}: {step_data.get('agent')}")
        self.geometry("900x600")
        self.editing_data = copy.deepcopy(step_data); self.agent_def = agent_def or {}
        self.saved = False; self.param_entries = {}
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        self.create_widgets(); self.transient(parent); self.grab_set()

    def create_widgets(self):
        toolbox = ctk.CTkFrame(self, width=250); toolbox.grid(row=0, column=0, padx=10, pady=10, sticky="ns")
        ctk.CTkLabel(toolbox, text="Optional Inputs", font=ctk.CTkFont(weight="bold")).pack(pady=5)
        self.toolbox_scroll = ctk.CTkScrollableFrame(toolbox, fg_color="transparent"); self.toolbox_scroll.pack(fill="both", expand=True)
        form = ctk.CTkFrame(self); form.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        form.grid_rowconfigure(0, weight=1); form.grid_columnconfigure(0, weight=1)
        self.params_frame = ctk.CTkScrollableFrame(form, label_text="Parameters & Outputs"); self.params_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")
        button_frame = ctk.CTkFrame(self); button_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save", command=self.save, fg_color="green").pack(side="right", padx=10)
        self.refresh_all_forms()

    def refresh_all_forms(self): self.refresh_params_form(); self.refresh_toolbox()
    
    def refresh_toolbox(self):
        for widget in self.toolbox_scroll.winfo_children(): widget.destroy()
        master_optionals = set(self.agent_def.get("optional_inputs", []))
        current_params = set(self.editing_data.get("params", {}).keys())
        available = sorted(list(master_optionals - current_params))
        for param in available: ctk.CTkButton(self.toolbox_scroll, text=f"+ {param}", fg_color="gray", command=lambda p=param: self.add_param_from_toolbox(p)).pack(fill="x", padx=5, pady=2)

    def refresh_params_form(self):
        for widget in self.params_frame.winfo_children(): widget.destroy()
        self.param_entries.clear()
        ctk.CTkLabel(self.params_frame, text="Outputs", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
        for i, item in enumerate(self.editing_data.setdefault('output', [])):
            entry = ctk.CTkEntry(self.params_frame); entry.insert(0, item); entry.pack(fill="x", padx=5, pady=2); self.param_entries[f'output_{i}'] = entry
        
        ctk.CTkLabel(self.params_frame, text="Parameters (Key = Value)", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(10,0))
        params = self.editing_data.setdefault("params", {})
        for key, value in list(params.items()):
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent"); param_frame.pack(fill="x", pady=2)
            is_required = key in self.agent_def.get("inputs", [])
            if is_required: ctk.CTkLabel(param_frame, text=key, width=200).pack(side="left", padx=5)
            else: key_entry = ctk.CTkEntry(param_frame, width=200); key_entry.insert(0, key); key_entry.pack(side="left", padx=5); self.param_entries[f'param_key_{key}'] = key_entry
            value_entry = ctk.CTkEntry(param_frame); value_entry.insert(0, value); value_entry.pack(side="left", padx=5, expand=True, fill="x")
            if not is_required:
                remove_btn = ctk.CTkButton(param_frame, text="X", width=30, fg_color="#D32F2F", hover_color="#B71C1C", command=lambda k=key: self.remove_param(k)); remove_btn.pack(side="left", padx=5)
            self.param_entries[f'param_val_{key}'] = value_entry
        ctk.CTkButton(self.params_frame, text="+ Add Parameter", command=self.add_param).pack(pady=10, anchor="w", padx=5)

    def add_param(self):
        i = 0
        while f"new_param_{i}" in self.editing_data.get("params", {}): i+=1
        self.editing_data.setdefault("params", {})[f"new_param_{i}"] = "new_value"; self.refresh_all_forms()
    def add_param_from_toolbox(self, param_name): self.editing_data.setdefault("params", {})[param_name] = ""; self.refresh_all_forms()
    def remove_param(self, key):
        if key in self.editing_data.get("params", {}): del self.editing_data["params"][key]
        self.refresh_all_forms()

    def get_form_data(self):
        new_outputs = []; new_params = {}
        original_output_len = len(self.editing_data.get('output',[]))
        original_param_keys = list(self.editing_data.get("params", {}).keys())
        for i in range(original_output_len): new_outputs.append(self.param_entries[f'output_{i}'].get())
        for key in original_param_keys:
            is_required = key in self.agent_def.get("inputs", [])
            new_key = key if is_required else self.param_entries[f'param_key_{key}'].get()
            new_val = self.param_entries[f'param_val_{key}'].get()
            if not new_key: continue
            new_params[new_key] = new_val
        self.editing_data['output'] = new_outputs; self.editing_data['params'] = new_params
        
    def save(self): self.get_form_data(); self.saved = True; self.destroy()
    def cancel(self): self.saved = False; self.destroy()
    def get_result(self): return self.editing_data

if __name__ == "__main__":
    ctk.set_appearance_mode("System"); ctk.set_default_color_theme("blue"); app = App(); app.mainloop()