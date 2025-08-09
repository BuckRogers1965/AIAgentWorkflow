# --- START OF FILE ui_step_editor.py ---
import customtkinter as ctk
import copy
import re

class StepEditorModal(ctk.CTkToplevel):
    def __init__(self, parent, index, step_data, agent_def, parent_workflow_data, theme):
        super().__init__(parent)
        self.title(f"Edit Step {index}: {step_data.get('agent')}")
        self.geometry("1100x700")
        self.theme = theme
        self.agent_def = agent_def or {}
        self.editing_data = copy.deepcopy(step_data)
        self.step_index = index
        self.parent_workflow_data = parent_workflow_data or {"inputs": [], "optional_inputs": [], "steps": []}
        self.active_entry = None
        self.available_vars = set()
        
        dummy_entry = ctk.CTkEntry(self)
        self.default_border_color = dummy_entry.cget("border_color")
        self.literal_border_color = self.theme['colors']['accent_primary']
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

    def create_widgets(self):
        self.configure(fg_color=self.theme['colors']['bg_primary'])
        
        selector_frame = ctk.CTkFrame(self, width=300, fg_color=self.theme['colors']['bg_secondary'])
        selector_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        selector_frame.grid_rowconfigure(0, weight=1); selector_frame.grid_columnconfigure(0, weight=1)
        
        selector_tabs = ctk.CTkTabview(selector_frame, fg_color=self.theme['colors']['bg_tertiary'])
        selector_tabs.grid(row=0, column=0, sticky="nsew")
        
        self.wf_req_inputs_frame = ctk.CTkScrollableFrame(selector_tabs.add("Workflow Inputs"), label_text="Required")
        self.wf_req_inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.wf_opt_inputs_frame = ctk.CTkScrollableFrame(selector_tabs.tab("Workflow Inputs"), label_text="Optional")
        self.wf_opt_inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.step_outputs_frame = ctk.CTkScrollableFrame(selector_tabs.add("Step Outputs"), label_text="Available Outputs")
        self.step_outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        form = ctk.CTkFrame(self, fg_color=self.theme['colors']['bg_secondary'])
        form.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        form.grid_rowconfigure(0, weight=1); form.grid_columnconfigure(0, weight=1)
        
        self.params_frame = ctk.CTkScrollableFrame(form, label_text="Parameters & Outputs")
        self.params_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save", command=self.save, fg_color=self.theme['colors']['success']).pack(side="right", padx=10)

        self.populate_variable_selector()
        self.refresh_params_form()

    def refresh_params_form(self):
        for widget in self.params_frame.winfo_children(): widget.destroy()
        self.param_entries.clear()
        
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'], weight="bold")
        param_font = ctk.CTkFont(family=self.theme['fonts']['editor_step_param_family'], size=self.theme['fonts']['editor_step_param_size'])
        
        required_inputs = set(self.agent_def.get("inputs", []))
        optional_inputs = set(self.agent_def.get("optional_inputs", []))
        
        self.editing_data.setdefault('params', {})
        self.editing_data.setdefault('output', [])

        ctk.CTkLabel(self.params_frame, text="Outputs", font=label_font).pack(anchor="w", padx=5)
        for i, item in enumerate(self.editing_data['output']):
            entry = ctk.CTkEntry(self.params_frame, font=param_font, fg_color=self.theme['colors']['editor_step_param_bg'])
            entry.insert(0, item)
            entry.pack(fill="x", padx=5, pady=2)
            self.param_entries[f'output_{i}'] = entry

        if required_inputs:
            ctk.CTkLabel(self.params_frame, text="Required Parameters", font=label_font).pack(anchor="w", padx=5, pady=(15, 0))
        for key in sorted(list(required_inputs)):
            if key not in self.editing_data["params"]: self.editing_data["params"][key] = f"${key}"
            
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent"); param_frame.pack(fill="x", pady=2)
            ctk.CTkLabel(param_frame, text=key, width=200, font=param_font).pack(side="left", padx=5)
            value_entry = ctk.CTkEntry(param_frame, font=param_font, fg_color=self.theme['colors']['editor_step_param_bg'])
            value_entry.insert(0, self.editing_data["params"].get(key, ""))
            value_entry.pack(side="left", padx=5, expand=True, fill="x")
            
            value_entry.bind("<FocusIn>", lambda event, entry=value_entry: self.set_active_entry(entry))
            value_entry.bind("<KeyRelease>", lambda event, entry=value_entry: self.validate_entry_variables(entry))
            
            self.param_entries[f'param_val_{key}'] = value_entry
            self.validate_entry_variables(value_entry)

        optional_and_custom_keys = set(self.editing_data["params"].keys()) - required_inputs
        if optional_and_custom_keys:
            ctk.CTkLabel(self.params_frame, text="Optional / Custom Parameters", font=label_font).pack(anchor="w", padx=5, pady=(15, 0))
        for key in sorted(list(optional_and_custom_keys)):
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent"); param_frame.pack(fill="x", pady=2)
            
            if key in optional_inputs:
                ctk.CTkLabel(param_frame, text=key, width=200, font=param_font).pack(side="left", padx=5)
            else:
                key_entry = ctk.CTkEntry(param_frame, width=200, font=param_font, fg_color=self.theme['colors']['editor_step_param_bg'])
                key_entry.insert(0, key); key_entry.pack(side="left", padx=5)
                self.param_entries[f'param_key_{key}'] = key_entry

            value_entry = ctk.CTkEntry(param_frame, font=param_font, fg_color=self.theme['colors']['editor_step_param_bg'])
            value_entry.insert(0, self.editing_data["params"].get(key, ""))
            value_entry.pack(side="left", padx=5, expand=True, fill="x")
            
            value_entry.bind("<FocusIn>", lambda event, entry=value_entry: self.set_active_entry(entry))
            value_entry.bind("<KeyRelease>", lambda event, entry=value_entry: self.validate_entry_variables(entry))

            self.param_entries[f'param_val_{key}'] = value_entry
            self.validate_entry_variables(value_entry)
            
            remove_btn = ctk.CTkButton(param_frame, text="X", width=30, fg_color=self.theme['colors']['error'], command=lambda k=key: self.remove_param(k))
            remove_btn.pack(side="left", padx=5)
        
        available_options = sorted(list(optional_inputs - set(self.editing_data["params"].keys())))
        if available_options:
            option_menu = ctk.CTkOptionMenu(self.params_frame, values=["Add Optional Input..."] + available_options, command=self.add_optional_param)
            option_menu.pack(pady=10, padx=5, anchor="w")
    
    # ... (rest of StepEditorModal is unchanged) ...
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
        if not text:
            entry_widget.configure(border_color=self.default_border_color, border_width=1)
            return
        if not found_vars:
            entry_widget.configure(border_color=self.literal_border_color, border_width=2)
            return
        
        is_valid = all(var in self.available_vars for var in found_vars)
        entry_widget.configure(border_color="red" if not is_valid else self.default_border_color, border_width=2 if not is_valid else 1)

    def set_active_entry(self, entry_widget):
        self.active_entry = entry_widget

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
            for output_var in step.get('output', []):
                create_var_button(self.step_outputs_frame, output_var)
    def add_optional_param(self, param_name: str):
        if param_name and "Add Optional" not in param_name:
            self.editing_data.setdefault("params", {})[param_name] = ""
            self.refresh_params_form()
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
            if new_key:
                new_params[new_key] = self.param_entries[f'param_val_{key_ref}'].get()
        self.editing_data['params'] = new_params
        
    def save(self):
        self.get_form_data()
        self.saved = True
        self.destroy()

    def cancel(self):
        self.saved = False
        self.destroy()

    def get_result(self):
        return self.editing_data