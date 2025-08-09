# --- START OF FILE ui_run_modal.py ---
import customtkinter as ctk
import json
import copy
import io
import logging
import base64
import re
from functools import reduce 
import operator
from tkinter import messagebox

def get_nested(data, key_str):
    try:
        return reduce(operator.getitem, key_str.split('.'), data)
    except (KeyError, TypeError, AttributeError):
        return None

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            try: return o.decode('utf-8')
            except UnicodeDecodeError: return f"<base64_encoded_bytes>{base64.b64encode(o).decode('utf-8')}</base64_encoded_bytes>"
        return json.JSONEncoder.default(self, o)

class RunAgentModal(ctk.CTkToplevel):
    def __init__(self, parent_app, agent_name, agent_data, core_lib=None, theme=None):
        super().__init__(parent_app)
        self.title(f"Run Agent: {agent_name}")
        self.geometry("900x700")
        self.app = parent_app
        self.agent_name = agent_name
        self.agent_data = agent_data
        
        self.core_lib = core_lib
        # --- ROBUST THEME HANDLING ---
        # If the theme is missing, create an empty dict to prevent crashes
        self.theme = theme if theme is not None else {}
        
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

    def _get_theme_color(self, key, default_color):
        """Safely gets a color from the theme, providing a default."""
        return self.theme.get("colors", {}).get(key, default_color)

    def _get_theme_font(self, key, default_font):
        """Safely gets a font tuple from the theme, providing a default."""
        font_info = self.theme.get("fonts", {})
        family = font_info.get(f"{key}_family", default_font[0])
        size = font_info.get(f"{key}_size", default_font[1])
        return (family, size)

    def close(self):
        self.save_current_run_config()
        self.destroy()

    def create_widgets(self):
        self.configure(fg_color=self._get_theme_color('bg_primary', '#242424'))

        input_container = ctk.CTkFrame(self, fg_color=self._get_theme_color('bg_secondary', '#2B2B2B'))
        input_container.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        input_container.grid_columnconfigure(0, weight=1)
        self.inputs_scroll_frame = ctk.CTkScrollableFrame(input_container, label_text="Inputs", fg_color=self._get_theme_color('bg_primary', '#242424'))
        self.inputs_scroll_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
        button_frame = ctk.CTkFrame(input_container, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=(5,10))
        ctk.CTkLabel(button_frame, text="Log Level:").pack(side="left", padx=(0,5))
        self.log_level_var = ctk.StringVar(value="INFO")
        log_level_menu = ctk.CTkOptionMenu(button_frame, variable=self.log_level_var, values=["DEBUG", "INFO", "WARNING", "ERROR"])
        log_level_menu.pack(side="left")
        help_button = ctk.CTkButton(button_frame, text="?", width=25, command=self.show_test_help)
        help_button.pack(side="left", padx=10)
        run_button = ctk.CTkButton(button_frame, text="Run Agent & Validate", command=self.execute_run, fg_color=self._get_theme_color('success', 'green'))
        run_button.pack(side="right")
        
        output_container = ctk.CTkFrame(self, fg_color=self._get_theme_color('bg_secondary', '#2B2B2B'))
        output_container.grid(row=1, column=0, padx=10, pady=(0, 10), sticky="nsew")
        output_container.grid_columnconfigure(0, weight=1)
        output_container.grid_rowconfigure(0, weight=1)
        self.tab_view = ctk.CTkTabview(output_container, fg_color=self._get_theme_color('bg_tertiary', '#323232'))
        self.tab_view.pack(fill="both", expand=True)
        
        self.output_tab = self.tab_view.add("Final Output")
        self.log_tab = self.tab_view.add("Verbose Log")
        self.test_tab = self.tab_view.add("Unit Tests")
        
        # --- ROBUST FONT AND COLOR USAGE ---
        code_font = self._get_theme_font('code', ("Courier", 12))
        self.output_textbox = ctk.CTkTextbox(self.output_tab, font=code_font, wrap="word")
        self.output_textbox.pack(fill="both", expand=True)
        self.log_textbox = ctk.CTkTextbox(self.log_tab, font=code_font, wrap="word")
        self.log_textbox.pack(fill="both", expand=True)
        # --- END OF FIX ---
        
        self.create_test_tab_widgets()
        self.populate_input_fields()

    def add_input_field(self, parent, key, is_optional=True):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", expand=True, pady=2)
        ctk.CTkLabel(frame, text=key, width=150).pack(side="left", padx=5)
        entry = ctk.CTkEntry(frame); entry.pack(side="left", fill="x", expand=True, padx=5)
        self.input_entries[key] = entry
        if is_optional:
            remove_btn = ctk.CTkButton(frame, text="X", width=30, fg_color=self._get_theme_color('error', 'red'), command=lambda k=key: self.remove_optional_field(k))
            remove_btn.pack(side="left", padx=5)
            self.optional_frames[key] = frame

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
        remove_btn = ctk.CTkButton(frame, text="X", width=30, fg_color=self._get_theme_color('error', 'red'), command=lambda f=frame: f.destroy())
        remove_btn.grid(row=0, column=4, padx=5, pady=5)
        self.test_widgets.append({ "frame": frame, "var_menu": var_menu, "assertion_menu": assertion_menu, "value_entry": value_entry, "result_label": result_label })
        
    def execute_run(self):
        self.save_current_run_config()
        
        if not self.core_lib:
            messagebox.showerror("Error", "Core workflow engine is not loaded. Cannot run agent.")
            return

        self.core_lib["dynamic_workflows_agents"].log_text_limit = int(self.app.config_manager.config.get('workflow_settings', {}).get('log_text_limit', 500))
        
        workflow_inputs = {key: entry.get() for key, entry in self.input_entries.items()}
        temp_config = copy.deepcopy(self.app.config_manager.config)
        temp_config['agents'][self.agent_name] = self.agent_data
        
        self.core_lib["setup_depth_manager"](temp_config)
        agent_to_run = self.agent_data
        
        if not agent_to_run: 
            messagebox.showerror("Error", "Could not prepare agent for execution.")
            return

        log_stream = io.StringIO()
        ui_log_handler = logging.StreamHandler(log_stream)
        ui_log_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
        root_logger.addHandler(ui_log_handler)
        root_logger.setLevel(getattr(logging, self.log_level_var.get(), logging.INFO))
        
        final_result_tape, final_status = {}, {"status": {"value": -99, "reason": "Execution did not run"}}
        try:
            final_result_tape, final_status = self.core_lib["exec_agent"](
                agent=agent_to_run, 
                agent_name=self.agent_name, 
                config=temp_config, 
                cli_args=workflow_inputs, 
                results={},
                force_recompile=True
            )
        except Exception as e:
            final_result_tape = {"__error__": "An unhandled exception occurred during workflow execution.", "details": str(e)}
            logging.exception("Workflow execution failed")

        for textbox in [self.log_textbox, self.output_textbox]: textbox.configure(state="normal"); textbox.delete("1.0", "end")
        self.log_textbox.insert("1.0", log_stream.getvalue())
        self.output_textbox.insert("1.0", json.dumps(final_result_tape, indent=2, cls=CustomJSONEncoder))
        for textbox in [self.log_textbox, self.output_textbox]: textbox.configure(state="disabled")
        
        self.run_assertions(final_result_tape, final_status)
        self.tab_view.set("Unit Tests")
        
    # ... (The rest of the class is unchanged and should now work) ...
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
        help_content = """
**Agent Unit Testing Framework**
...
"""
        self.app.show_help_modal("Unit Testing Help", help_content)
