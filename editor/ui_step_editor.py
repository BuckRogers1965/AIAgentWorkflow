# --- START OF FILE ui_step_editor.py ---

import customtkinter as ctk
import copy
import re
import webbrowser

from ui_utils import get_nested, ToolTip, ValidationMixin, PresetSelectorMixin, ThemeMixin

class StepEditorModal(ctk.CTkToplevel, ValidationMixin, PresetSelectorMixin, ThemeMixin):
    def __init__(self, parent, index, step_data, agent_def, parent_workflow_data, theme):
        super().__init__(parent)
        self.app_ref = parent
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
        self.literal_border_color = self.get_theme_color('accent_primary', '#0078D4')
        dummy_entry.destroy()

        # --- NEW: Logic to find sibling versions ---
        self.sibling_versions = []
        self.current_agent_name = self.editing_data.get('agent', '')
        
        version_match = re.match(r'^(.*)/v(\d+)\.\d+$', self.current_agent_name)
        if version_match:
            base_path = version_match.group(1)
            major_version = version_match.group(2)
            
            all_agents = self.app_ref.config_manager.get_agent_names()
            
            for agent_name in all_agents:
                if agent_name.startswith(f"{base_path}/v{major_version}."):
                    self.sibling_versions.append(agent_name)
            
            self.sibling_versions.sort(key=lambda v: [int(x) for x in re.search(r'v(\d+)\.(\d+)$', v).groups()])
        # --- END NEW ---

        self.calculate_available_variables()
        self.saved = False
        self.param_entries = {}
        self.tooltips = {}

        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=2)
        self.grid_rowconfigure(0, weight=1)

        self.create_widgets()
        self.transient(parent)
        self.grab_set()

    def create_widgets(self):
        # --- START of logic to find sibling versions ---
        sibling_versions = []
        current_agent_name = self.editing_data.get('agent', '')
        
        version_match = re.match(r'^(.*)/v(\d+)\.\d+$', current_agent_name)
        if version_match:
            base_path = version_match.group(1)
            major_version = version_match.group(2)
            
            all_agents = self.app_ref.config_manager.get_agent_names()
            
            for agent_name in all_agents:
                if agent_name.startswith(f"{base_path}/v{major_version}."):
                    sibling_versions.append(agent_name)
            
            sibling_versions.sort(key=lambda v: [int(x) for x in re.search(r'v(\d+)\.(\d+)$', v).groups()])
        # --- END of logic ---

        self.configure(fg_color=self.get_theme_color('bg_primary', '#242424'))
        
        selector_frame = ctk.CTkFrame(self, width=300, fg_color=self.get_theme_color('bg_secondary', '#2B2B2B'))
        selector_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        selector_frame.grid_rowconfigure(0, weight=1)
        selector_frame.grid_columnconfigure(0, weight=1)
        
        selector_tabs = ctk.CTkTabview(selector_frame, fg_color=self.get_theme_color('bg_tertiary', '#323232'))
        selector_tabs.grid(row=0, column=0, sticky="nsew")
        
        self.wf_req_inputs_frame = ctk.CTkScrollableFrame(selector_tabs.add("Workflow Inputs"), label_text="Required")
        self.wf_req_inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.wf_opt_inputs_frame = ctk.CTkScrollableFrame(selector_tabs.tab("Workflow Inputs"), label_text="Optional")
        self.wf_opt_inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        self.step_outputs_frame = ctk.CTkScrollableFrame(selector_tabs.add("Step Outputs"), label_text="Available Outputs")
        self.step_outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        form = ctk.CTkFrame(self, fg_color=self.get_theme_color('bg_secondary', '#2B2B2B'))
        form.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        form.grid_rowconfigure(1, weight=1)
        form.grid_columnconfigure(0, weight=1)
        
        form_header = ctk.CTkFrame(form, fg_color="transparent")
        form_header.grid(row=0, column=0, sticky="ew", padx=5, pady=(5,0))

        if sibling_versions:
            ctk.CTkLabel(form_header, text="Version:").pack(side="left", padx=(0, 5))
            # Create a StringVar to hold the dropdown's current value
            self.agent_version_var = ctk.StringVar(value=current_agent_name)
            self.version_menu = ctk.CTkOptionMenu(
                form_header,
                values=sibling_versions,
                variable=self.agent_version_var
                # NO 'command' ATTRIBUTE
            )
            self.version_menu.pack(side="left")
        
        self.params_frame = ctk.CTkScrollableFrame(form, label_text="Parameters & Outputs")
        self.params_frame.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save", command=self.save, 
                     fg_color=self.get_theme_color('success', 'green')).pack(side="right", padx=10)

        self.populate_variable_selector()
        self.refresh_params_form()

    def on_version_change(self, new_version_name: str):
        """Callback when a new version is selected from the dropdown."""
        if new_version_name == self.current_agent_name:
            return

        from tkinter import messagebox
        if not messagebox.askyesno("Change Version?", f"Change this step to use version '{new_version_name}'?\n\nThis will reset the parameters below to match the new version's definition."):
            self.version_menu.set(self.current_agent_name) # Revert dropdown
            return

        self.current_agent_name = new_version_name
        self.editing_data['agent'] = new_version_name

        new_agent_def = self.app_ref.config_manager.get_agent_data(new_version_name)
        if not new_agent_def:
            messagebox.showerror("Error", f"Could not load definition for {new_version_name}")
            return

        self.agent_def = new_agent_def
        
        # Reset params and outputs to the new agent's defaults
        self.editing_data['params'] = {key: f"${key}" for key in self.agent_def.get("inputs", [])}
        self.editing_data['output'] = self.agent_def.get("outputs", []).copy()
        
        self.title(f"Edit Step {self.step_index}: {new_version_name}")
        self.refresh_params_form()

    # --- NO OTHER CHANGES ARE NEEDED BELOW THIS LINE ---

    def save(self):
        self.get_form_data()
        self.saved = True
        self.destroy()

    def cancel(self):
        self.saved = False
        self.destroy()

    def get_result(self):
        return self.editing_data

    def _get_param_font(self):
        return ctk.CTkFont(
            family=self.get_theme_font('editor_step_param', ('Courier', 12))[0],
            size=self.get_theme_font('editor_step_param', ('Courier', 12))[1]
        )

    def _get_entry_bg_color(self):
        return self.get_theme_color('editor_step_param_bg', 'transparent')

    def refresh_params_form(self):
        for widget in self.params_frame.winfo_children(): 
            widget.destroy()
        self.param_entries.clear()
        
        label_font = ctk.CTkFont(
            family=self.get_theme_font('main', ('Arial', 12))[0],
            size=self.get_theme_font('label', ('Arial', 12))[1],
            weight="bold"
        )
        
        required_inputs = set(self.agent_def.get("inputs", []))
        optional_inputs = set(self.agent_def.get("optional_inputs", []))
        all_defined_inputs = required_inputs | optional_inputs
        
        self.editing_data.setdefault('params', {})
        self.editing_data.setdefault('output', [])

        param_hints = self.agent_def.get("gui", {}).get("param_hints", {})

        def create_param_widget(parent, key, value):
            hint = param_hints.get(key, {})
            if hint.get("widget_type") == "preset_selector":
                self.create_preset_selector_param_widget(parent, key, value, hint)
            else:
                self.create_standard_param_widget(parent, key, value, hint)

        ctk.CTkLabel(self.params_frame, text="Outputs", font=label_font).pack(anchor="w", padx=5)
        for i, item in enumerate(self.editing_data['output']):
            param_font = self._get_param_font()
            entry = ctk.CTkEntry(self.params_frame, font=param_font, 
                               fg_color=self._get_entry_bg_color())
            entry.insert(0, item)
            entry.pack(fill="x", padx=5, pady=2)
            entry.bind("<FocusIn>", lambda event, entry=entry: self.set_active_entry(entry))
            self.param_entries[f'output_{i}'] = entry

        if required_inputs:
            ctk.CTkLabel(self.params_frame, text="Required Parameters", 
                        font=label_font).pack(anchor="w", padx=5, pady=(15, 0))
        for key in sorted(list(required_inputs)):
            if key not in self.editing_data["params"]: 
                self.editing_data["params"][key] = f"${key}"
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
            param_frame.pack(fill="x", pady=2, padx=5)
            create_param_widget(param_frame, key, self.editing_data["params"].get(key, ""))
        
        all_step_param_keys = set(self.editing_data["params"].keys())
        orphaned_keys = all_step_param_keys - all_defined_inputs
        if orphaned_keys:
            param_font = self._get_param_font()
            ctk.CTkLabel(self.params_frame, text="Orphaned Parameters (No longer valid for this agent)", 
                         font=label_font, text_color=self.get_theme_color('error', 'red')).pack(anchor="w", padx=5, pady=(15, 0))
        
            for key in sorted(list(orphaned_keys)):
                value = self.editing_data["params"].get(key, "")
                param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
                param_frame.pack(fill="x", pady=2, padx=5)
                
                label = ctk.CTkLabel(param_frame, text=key, width=200, font=param_font, text_color="gray")
                label.pack(side="left")
                
                entry = ctk.CTkEntry(param_frame, font=param_font, border_color="red", border_width=2)
                entry.insert(0, value)
                entry.configure(state="disabled")
                entry.pack(side="left", expand=True, fill="x")

                remove_btn = ctk.CTkButton(param_frame, text="X", width=30, 
                                         fg_color=self.get_theme_color('error', 'red'), 
                                         command=lambda k=key: self.remove_param(k))
                remove_btn.pack(side="left", padx=5)

        optional_and_custom_keys = set(self.editing_data["params"].keys()) - required_inputs - orphaned_keys
        if optional_and_custom_keys:
            ctk.CTkLabel(self.params_frame, text="Optional / Custom Parameters", 
                        font=label_font).pack(anchor="w", padx=5, pady=(15, 0))
        for key in sorted(list(optional_and_custom_keys)):
            param_frame = ctk.CTkFrame(self.params_frame, fg_color="transparent")
            param_frame.pack(fill="x", pady=2, padx=5)
            param_font = self._get_param_font()
            
            key_container = ctk.CTkFrame(param_frame, fg_color="transparent")
            key_container.pack(side="left", padx=5)
            if key in optional_inputs:
                ctk.CTkLabel(key_container, text=key, width=200, font=param_font).pack(side="left")
            else:
                key_entry = ctk.CTkEntry(key_container, width=200, font=param_font, 
                                       fg_color=self._get_entry_bg_color())
                key_entry.insert(0, key)
                key_entry.pack(side="left")
                key_entry.bind("<FocusIn>", lambda event, entry=key_entry: self.set_active_entry(entry))
                self.param_entries[f'param_key_{key}'] = key_entry
            
            value_container = ctk.CTkFrame(param_frame, fg_color="transparent")
            value_container.pack(side="left", expand=True, fill="x")
            create_param_widget(value_container, key, self.editing_data["params"].get(key, ""))
            
            remove_btn = ctk.CTkButton(param_frame, text="X", width=30, 
                                     fg_color=self.get_theme_color('error', 'red'), 
                                     command=lambda k=key: self.remove_param(k))
            remove_btn.pack(side="left", padx=5)
        
        available_options = sorted(list(optional_inputs - set(self.editing_data["params"].keys())))
        if available_options:
            option_menu = ctk.CTkOptionMenu(self.params_frame, 
                                          values=["Add Optional Input..."] + available_options, 
                                          command=self.add_optional_param)
            option_menu.pack(pady=10, padx=5, anchor="w")

    def create_standard_param_widget(self, parent, key, value, hint):
        param_font = self._get_param_font()
        if not (parent.winfo_children() and isinstance(parent.winfo_children()[0], ctk.CTkLabel)):
            ctk.CTkLabel(parent, text=key, width=200, font=param_font).pack(side="left")

        value_entry = ctk.CTkEntry(parent, font=param_font, fg_color=self._get_entry_bg_color())
        value_entry.insert(0, value)
        value_entry.pack(side="left", expand=True, fill="x")
        value_entry.bind("<FocusIn>", lambda event, entry=value_entry: self.set_active_entry(entry))
        value_entry.bind("<KeyRelease>", lambda event, entry=value_entry: self.validate_entry_variables(entry))
        self.param_entries[f'param_val_{key}'] = value_entry
        self.apply_common_param_hints(key, value_entry, hint)
        self.validate_entry_variables(value_entry)

    def create_preset_selector_param_widget(self, parent, key, value, hint):
        param_font = self._get_param_font()
        if not (parent.winfo_children() and isinstance(parent.winfo_children()[0], ctk.CTkLabel)):
            ctk.CTkLabel(parent, text=key, width=200, font=param_font).pack(side="left")
        
        widget_frame = ctk.CTkFrame(parent, fg_color="transparent")
        widget_frame.pack(side="left", fill="x", expand=True)

        data_source_path = hint.get("data_source")
        preset_data = get_nested(self.app_ref.config_manager.config, data_source_path) or {}
        preset_names = ["Custom"] + sorted(preset_data.keys())
        
        value_entry = ctk.CTkEntry(widget_frame, font=param_font, fg_color=self._get_entry_bg_color())
        value_entry.insert(0, value)
        
        option_menu = ctk.CTkOptionMenu(widget_frame, values=preset_names, width=150, font=param_font)
        option_menu.pack(fill="x", expand=True)
        value_entry.pack(fill="x", expand=True, pady=(2, 0))
        
        value_entry.bind("<FocusIn>", lambda event, entry=value_entry: self.set_active_entry(entry))
        self.param_entries[f'param_val_{key}'] = value_entry
        self.apply_common_param_hints(key, value_entry, hint)

        def on_menu_select(choice):
            if choice != "Custom":
                selected_value = preset_data.get(choice, "")
                value_entry.delete(0, "end")
                value_entry.insert(0, selected_value)
            self.validate_entry_variables(value_entry)
        
        def on_entry_change(*args):
            current_text = value_entry.get()
            self.validate_entry_variables(value_entry)
            if not current_text.startswith('$'):
                matching_preset = "Custom"
                for name, p_value in preset_data.items():
                    if current_text == p_value:
                        matching_preset = name
                        break
                option_menu.set(matching_preset)

        option_menu.configure(command=on_menu_select)
        value_entry.bind("<KeyRelease>", on_entry_change)
        self.validate_entry_variables(value_entry)
        on_entry_change()

    def calculate_available_variables(self):
        self.available_vars.clear()
        self.available_vars.add("step_index")
        for var in self.parent_workflow_data.get('inputs', []): 
            self.available_vars.add(var)
        for var in self.parent_workflow_data.get('optional_inputs', []): 
            self.available_vars.add(var)
        for i in range(self.step_index):
            step = self.parent_workflow_data['steps'][i]
            for output_var in step.get('output', []): 
                self.available_vars.add(output_var)

    def validate_entry_variables(self, entry_widget):
        text = entry_widget.get()
        found_vars = re.findall(r'\$(\w+)', text)
        if text.startswith('$'):
            if not found_vars:
                entry_widget.configure(border_color="red", border_width=2)
                return
            is_valid = all(var in self.available_vars for var in found_vars)
            entry_widget.configure(border_color="red" if not is_valid else self.default_border_color, border_width=2 if not is_valid else 1)
            return
        for key, widget in self.param_entries.items():
            if widget == entry_widget:
                param_key = key.replace("param_val_", "").replace("output_", "")
                param_hints = self.agent_def.get("gui", {}).get("param_hints", {})
                if param_key in param_hints and "validation" in param_hints[param_key]:
                    self.validate_entry_with_feedback(param_key, entry_widget, param_hints[param_key], self.available_vars)
                    return
        if not text:
            entry_widget.configure(border_color=self.default_border_color, border_width=1)
            return
        entry_widget.configure(border_color=self.literal_border_color, border_width=2)

    def set_active_entry(self, entry_widget):
        self.active_entry = entry_widget

    def append_variable_to_active_entry(self, var_name):
        if self.active_entry:
            cursor_pos = self.active_entry.index(ctk.INSERT)
            current_text = self.active_entry.get()
            if (cursor_pos > 0 and current_text and current_text[cursor_pos-1] not in (' ', '(')):
                variable_string = f" ${var_name}"
            else:
                variable_string = f"${var_name}"
            self.active_entry.insert(cursor_pos, variable_string)
            self.active_entry.focus()
            self.validate_entry_variables(self.active_entry)

    def populate_variable_selector(self):
        def create_var_button(parent, var_name):
            btn = ctk.CTkButton(parent, text=var_name, anchor="w", fg_color="gray", command=lambda v=var_name: self.append_variable_to_active_entry(v))
            btn.pack(fill="x", padx=5, pady=2)
        for var in sorted(list(self.parent_workflow_data.get('inputs', []))): 
            create_var_button(self.wf_req_inputs_frame, var)
        for var in sorted(list(self.parent_workflow_data.get('optional_inputs', []))): 
            create_var_button(self.wf_opt_inputs_frame, var)
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
        if (key in self.editing_data.get("params", {}) and key not in set(self.agent_def.get("inputs", []))):
            del self.editing_data["params"][key]
        self.refresh_params_form()

    def get_form_data(self):
        # Update the agent name from the dropdown if it exists
        if hasattr(self, 'agent_version_var'):
            self.editing_data['agent'] = self.agent_version_var.get()

        # Update output data
        self.editing_data['output'] = [
            self.param_entries[f'output_{i}'].get() 
            for i in range(len(self.editing_data.get('output', [])))
        ]
        
        # Update parameter data
        new_params = {}
        all_rendered_keys = {
            k.replace('param_val_', '') 
            for k in self.param_entries 
            if k.startswith('param_val_')
        }
        
        for key_ref in all_rendered_keys:
            if f'param_key_{key_ref}' in self.param_entries:
                new_key = self.param_entries[f'param_key_{key_ref}'].get()
            else:
                new_key = key_ref
                
            if new_key:
                new_params[new_key] = self.param_entries[f'param_val_{key_ref}'].get()
                
        self.editing_data['params'] = new_params
