import customtkinter as ctk
import json
import copy
import io
import logging
import base64
import re
from tkinter import messagebox
import webbrowser
from datetime import datetime

from ui_utils import get_nested, ToolTip, ValidationMixin, PresetSelectorMixin, ThemeMixin

class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            try: return o.decode('utf-8')
            except UnicodeDecodeError: return f"<base64_encoded_bytes>{base64.b64encode(o).decode('utf-8')}</base64_encoded_bytes>"
        return json.JSONEncoder.default(self, o)

class TestResultsModal(ctk.CTkToplevel, ThemeMixin):
    def __init__(self, parent, results, theme):
        super().__init__(parent)
        self.title("Test Run Results")
        self.geometry("1000x700")
        self.theme = theme
        self.all_results = results

        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self.create_widgets()
        self.display_results()
        
        self.transient(parent)
        self.grab_set()

    def create_widgets(self):
        header = ctk.CTkFrame(self)
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=10, pady=10)
        self.summary_label = ctk.CTkLabel(header, text="", font=ctk.CTkFont(weight="bold"))
        self.summary_label.pack(side="left", padx=10)
        self.filter_var = ctk.BooleanVar(value=False)
        filter_check = ctk.CTkCheckBox(header, text="Show Failed Only", variable=self.filter_var, command=self.display_results)
        filter_check.pack(side="right", padx=10)

        self.test_list_frame = ctk.CTkScrollableFrame(self, width=300, label_text="Tests")
        self.test_list_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0,10))

        self.details_frame = ctk.CTkScrollableFrame(self, label_text="Details")
        self.details_frame.grid(row=1, column=1, sticky="nsew", padx=10, pady=(0,10))

    def display_results(self):
        for widget in self.test_list_frame.winfo_children(): widget.destroy()
        
        passed_count = sum(1 for r in self.all_results if r['passed'])
        total_count = len(self.all_results)
        self.summary_label.configure(text=f"Results: {passed_count} / {total_count} Tests Passed", text_color="green" if passed_count == total_count else "red")
        
        results_to_show = [r for r in self.all_results if not self.filter_var.get() or not r['passed']]

        for result in results_to_show:
            status_color = "green" if result['passed'] else "red"
            btn = ctk.CTkButton(self.test_list_frame, text=result['name'], fg_color=status_color, command=lambda r=result: self.show_details(r))
            btn.pack(fill="x", padx=5, pady=2)

    def show_details(self, result):
        for widget in self.details_frame.winfo_children(): widget.destroy()

        font_bold = ctk.CTkFont(weight="bold")
        font_code = self.get_theme_font('code', ("Courier", 12))

        ctk.CTkLabel(self.details_frame, text="Inputs Used:", font=font_bold).pack(anchor="w", pady=(5,2))
        inputs_text = json.dumps(result.get('inputs', {}), indent=2)
        inputs_box = ctk.CTkTextbox(self.details_frame, height=80, font=font_code)
        inputs_box.pack(fill="x", expand=True)
        inputs_box.insert("1.0", inputs_text)
        
        ctk.CTkLabel(self.details_frame, text="Assertions:", font=font_bold).pack(anchor="w", pady=(10,2))
        for assertion_result in result.get('assertion_results', []):
            status = "PASS" if assertion_result['passed'] else "FAIL"
            color = "green" if assertion_result['passed'] else "red"
            frame = ctk.CTkFrame(self.details_frame, border_width=1, border_color=color)
            frame.pack(fill="x", pady=2)
            ctk.CTkLabel(frame, text=status, text_color=color, width=50).pack(side="left", padx=5)
            ctk.CTkLabel(frame, text=assertion_result['text'], wraplength=500, justify="left").pack(side="left", padx=5, fill="x", expand=True)

        ctk.CTkLabel(self.details_frame, text="Final Output:", font=font_bold).pack(anchor="w", pady=(10,2))
        output_text = json.dumps(result.get('final_output', {}), indent=2, cls=CustomJSONEncoder)
        output_box = ctk.CTkTextbox(self.details_frame, height=150, font=font_code)
        output_box.pack(fill="x", expand=True)
        output_box.insert("1.0", output_text)

        ctk.CTkLabel(self.details_frame, text="Execution Log:", font=font_bold).pack(anchor="w", pady=(10,2))
        log_textbox = ctk.CTkTextbox(self.details_frame, height=200, font=font_code)
        log_textbox.insert("1.0", result.get('log', ''))
        log_textbox.pack(fill="x", expand=True)
        
        for widget in self.details_frame.winfo_children():
            if isinstance(widget, ctk.CTkTextbox):
                widget.configure(state="disabled")

class TestCaseEditorModal(ctk.CTkToplevel, ThemeMixin):
    def __init__(self, parent, test_case=None, agent_data=None, theme=None):
        super().__init__(parent)
        self.title("Edit Test Case")
        self.geometry("700x550")
        self.agent_data = agent_data or {}
        self.theme = theme
        
        if test_case is None:
            self.test_case = {"name": "", "notes": "", "date_added": datetime.now().isoformat(), "assertions": []}
        else:
            self.test_case = copy.deepcopy(test_case)
            self.test_case.setdefault("name", "")
            self.test_case.setdefault("notes", "")
            self.test_case.setdefault("date_added", datetime.now().isoformat())
            self.test_case.setdefault("assertions", [])
        
        self.saved = False
        self.create_widgets()
        self.transient(parent)
        self.grab_set()
        
    def create_widgets(self):
        header_frame = ctk.CTkFrame(self)
        header_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(header_frame, text="Test Name:").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.name_entry = ctk.CTkEntry(header_frame, width=300)
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=5)
        self.name_entry.insert(0, self.test_case["name"])
        
        ctk.CTkLabel(header_frame, text="Notes:").grid(row=1, column=0, sticky="nw", padx=5, pady=5)
        self.note_textbox = ctk.CTkTextbox(header_frame, height=60, width=300)
        self.note_textbox.grid(row=1, column=1, sticky="ew", padx=5, pady=5)
        self.note_textbox.insert("1.0", self.test_case["notes"])
        
        date_str = self.test_case["date_added"]
        try: formatted_date = datetime.fromisoformat(date_str).strftime("%Y-%m-%d %H:%M:%S")
        except: formatted_date = "Unknown"
        ctk.CTkLabel(header_frame, text=f"Created: {formatted_date}").grid(row=2, column=0, columnspan=2, sticky="w", padx=5, pady=5)
        
        header_frame.grid_columnconfigure(1, weight=1)
        
        assertions_container = ctk.CTkFrame(self)
        assertions_container.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        assertions_container.grid_columnconfigure(0, weight=1)
        assertions_container.grid_rowconfigure(1, weight=1)
        
        top_frame = ctk.CTkFrame(assertions_container)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(top_frame, text="Assertions", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(top_frame, text="+ Add Assertion", command=self.add_assertion).pack(side="right")
        
        self.assertions_frame = ctk.CTkScrollableFrame(assertions_container)
        self.assertions_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        button_frame = ctk.CTkFrame(self)
        button_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save", command=self.save, fg_color=self.get_theme_color('success', 'green')).pack(side="right", padx=10)
        
        for assertion in self.test_case["assertions"]: self.add_assertion(assertion)
    
    def add_assertion(self, assertion_data=None):
        if assertion_data is None:
            assertion_data = {"output_variable": "", "assertion_type": "Equals", "expected_value": ""}
            
        frame = ctk.CTkFrame(self.assertions_frame)
        frame.pack(fill="x", pady=2)
        frame.grid_columnconfigure(2, weight=1)
        
        output_vars = self.agent_data.get("outputs", []) + ["status.value"]
        var_menu = ctk.CTkOptionMenu(frame, values=["Select Variable"] + sorted(output_vars))
        var_menu.grid(row=0, column=0, padx=5, pady=5)
        var_menu.set(assertion_data.get("output_variable") or "Select Variable")
        
        assertion_menu = ctk.CTkOptionMenu(frame, values=["Equals", "Regex Match"])
        assertion_menu.grid(row=0, column=1, padx=5, pady=5)
        assertion_menu.set(assertion_data.get("assertion_type", "Equals"))
        
        value_entry = ctk.CTkEntry(frame, placeholder_text="Expected Value")
        value_entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        value_entry.insert(0, assertion_data.get("expected_value", ""))
        
        remove_btn = ctk.CTkButton(frame, text="X", width=30, fg_color=self.get_theme_color('error', 'red'), command=lambda f=frame: f.destroy())
        remove_btn.grid(row=0, column=3, padx=5, pady=5)
    
    def get_assertions(self):
        assertions = []
        for frame in self.assertions_frame.winfo_children():
            if not frame.winfo_exists(): continue
            children = frame.winfo_children()
            var_value = children[0].get()
            if var_value and var_value != "Select Variable":
                assertions.append({
                    "output_variable": var_value,
                    "assertion_type": children[1].get(),
                    "expected_value": children[2].get()
                })
        return assertions
    
    def save(self):
        self.test_case["name"] = self.name_entry.get()
        self.test_case["notes"] = self.note_textbox.get("1.0", "end-1c").strip()
        self.test_case["assertions"] = self.get_assertions()
        self.saved = True
        self.destroy()
        
    def cancel(self):
        self.saved = False
        self.destroy()

class RunAgentModal(ctk.CTkToplevel, ValidationMixin, PresetSelectorMixin, ThemeMixin):
    def __init__(self, parent_app, agent_name, agent_data, core_lib=None, theme=None):
        super().__init__(parent_app)
        self.title(f"Run Agent: {agent_name}")
        self.geometry("900x800")
        self.app = parent_app
        self.agent_name = agent_name
        self.agent_data = agent_data
        
        self.core_lib = core_lib
        self.theme = theme if theme is not None else {}
        
        self.input_entries = {}
        self.optional_frames = {}
        self.test_cases = []
        self.tooltips = {}
        self.current_test_case_index = None
        
        dummy_entry = ctk.CTkEntry(self)
        self.default_border_color = dummy_entry.cget("border_color")
        dummy_entry.destroy()

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)
        
        self.create_widgets()
        self.load_last_run_config()
        
        self.transient(parent_app)
        self.grab_set()
        self.protocol("WM_DELETE_WINDOW", self.close)

    def _get_param_font(self):
        return self.get_theme_font('code', ("Courier", 12))

    def _get_entry_bg_color(self):
        return self.get_theme_color('bg_primary', '#242424')

    def close(self):
        self.save_current_run_config()
        self.destroy()

    def create_widgets(self):
        self.configure(fg_color=self.get_theme_color('bg_primary', '#242424'))
        self.grid_rowconfigure(1, weight=1)

        container = ctk.CTkFrame(self, fg_color=self.get_theme_color('bg_secondary', '#2B2B2B'))
        container.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="nsew")
        container.grid_columnconfigure(0, weight=1)
        container.grid_rowconfigure(1, weight=1)
        
        top_section = ctk.CTkFrame(container, fg_color="transparent")
        top_section.grid(row=0, column=0, sticky="ew")
        top_section.grid_columnconfigure(0, weight=1)
        
        metadata_frame = ctk.CTkFrame(top_section)
        metadata_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        metadata_frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(metadata_frame, text="Test Name:", width=80).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        self.name_entry = ctk.CTkEntry(metadata_frame, placeholder_text="<Scratchpad - Unsaved>")
        self.name_entry.grid(row=0, column=1, sticky="ew", padx=5, pady=2)

        ctk.CTkLabel(metadata_frame, text="Notes:", width=80).grid(row=1, column=0, sticky="nw", padx=5, pady=2)
        self.notes_textbox = ctk.CTkTextbox(metadata_frame, height=50)
        self.notes_textbox.grid(row=1, column=1, sticky="ew", padx=5, pady=2)

        self.inputs_scroll_frame = ctk.CTkScrollableFrame(top_section, label_text="Inputs", 
                                                         fg_color=self.get_theme_color('bg_primary', '#242424'))
        self.inputs_scroll_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        button_frame = ctk.CTkFrame(top_section, fg_color="transparent")
        button_frame.grid(row=2, column=0, sticky="ew", padx=10, pady=(5,10))
        ctk.CTkLabel(button_frame, text="Log Level:").pack(side="left", padx=(0,5))
        self.log_level_var = ctk.StringVar(value="INFO")
        log_level_menu = ctk.CTkOptionMenu(button_frame, variable=self.log_level_var, 
                                         values=["DEBUG", "INFO", "WARNING", "ERROR"])
        log_level_menu.pack(side="left")
        self.run_button = ctk.CTkButton(button_frame, text="Run Current", command=self.execute_current_run)
        self.run_button.pack(side="right")
        self.save_test_button = ctk.CTkButton(button_frame, text="Save Test Case", command=self.save_test_case, fg_color=self.get_theme_color('accent_primary', 'blue'))
        self.save_test_button.pack(side="right", padx=5)
        self.toast_label = ctk.CTkLabel(button_frame, text="")
        self.toast_label.pack(side="right", padx=(0, 5))
        
        bottom_section = ctk.CTkTabview(container, fg_color=self.get_theme_color('bg_tertiary', '#323232'))
        bottom_section.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.output_tab = bottom_section.add("Last Run Output")
        self.log_tab = bottom_section.add("Last Run Log")
        self.test_tab = bottom_section.add("Saved Test Cases")
        self.assertions_tab = bottom_section.add("Assertions for Current Test")

        code_font = self.get_theme_font('code', ("Courier", 12))
        self.output_textbox = ctk.CTkTextbox(self.output_tab, font=code_font, wrap="word")
        self.output_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.log_textbox = ctk.CTkTextbox(self.log_tab, font=code_font, wrap="word")
        self.log_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        
        self.create_test_management_tab()
        self.create_assertions_tab()
        self.populate_input_fields()
    
    def show_toast(self, message):
        self.toast_label.configure(text=message, text_color=self.get_theme_color('success', 'green'))
        self.after(3000, lambda: self.toast_label.configure(text=""))

    def execute_current_run(self):
        inputs = {key: var.get() for key, var in self.input_entries.items()}
        result, status, log = self.run_agent_with_inputs(inputs)

        self.output_textbox.configure(state="normal")
        self.output_textbox.delete("1.0", "end")
        self.output_textbox.insert("1.0", json.dumps(result, indent=2, cls=CustomJSONEncoder))
        self.output_textbox.configure(state="disabled")
        
        self.log_textbox.configure(state="normal")
        self.log_textbox.delete("1.0", "end")
        self.log_textbox.insert("1.0", log)
        self.log_textbox.configure(state="disabled")
        
    def run_agent_with_inputs(self, inputs, run_name="Test"):
        if not self.core_lib: return {}, {}, "Core engine not loaded."

        # Get a fresh copy of the main config. This copy is stale.
        temp_config = copy.deepcopy(self.app.config_manager.config)

        # --- THIS IS THE FIX ---
        # The engine uses the agent's name to look up its definition from the 'config' object.
        # We MUST inject the live, fresh agent data from the editor (self.agent_data)
        # into this temporary config object before passing it to the engine.
        
        # self.agent_name is the name of the agent we are running.
        # self.agent_data is the fresh data dictionary from the editor, passed when the modal was created.
        temp_config['agents'][self.agent_name] = self.agent_data
        
        # --- END OF FIX ---

        self.core_lib["dynamic_workflows_agents"].log_text_limit = int(
            temp_config.get('workflow_settings', {}).get('log_text_limit', 500))

        log_stream = io.StringIO()
        handler = logging.StreamHandler(log_stream)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        logger = logging.getLogger()
        original_handlers = logger.handlers[:]
        logger.handlers = [handler]
        logger.setLevel(getattr(logging, self.log_level_var.get(), logging.INFO))
        
        result, status = {}, {}
        try:
            self.core_lib["setup_depth_manager"](temp_config)
            
            # Now, when exec_agent runs, it will use the temp_config that contains our fresh code.
            result, status = self.core_lib["exec_agent"](self.agent_data, self.agent_name, temp_config, inputs, {}, True)
            
        except Exception as e:
            logging.error(f"Execution failed for {run_name}: {e}", exc_info=True)
            status = {"status": {"value": 1, "reason": str(e)}}
        finally:
            logger.handlers = original_handlers

        return result, status, log_stream.getvalue()

    def create_test_management_tab(self):
        self.test_tab.grid_columnconfigure(0, weight=1)
        self.test_tab.grid_rowconfigure(1, weight=1)
        
        top_frame = ctk.CTkFrame(self.test_tab)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        
        ctk.CTkButton(top_frame, text="Load Scratchpad", command=self.load_scratchpad).pack(side="left", padx=5)
        ctk.CTkButton(top_frame, text="Run All Tests", command=self.run_all_tests).pack(side="right", padx=5)
        
        self.test_cases_frame = ctk.CTkScrollableFrame(self.test_tab, label_text="Saved Test Cases")
        self.test_cases_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def create_assertions_tab(self):
        self.assertions_tab.grid_columnconfigure(0, weight=1)
        self.assertions_tab.grid_rowconfigure(1, weight=1)

        top_frame = ctk.CTkFrame(self.assertions_tab)
        top_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(top_frame, text="Assertions", font=ctk.CTkFont(weight="bold")).pack(side="left")
        ctk.CTkButton(top_frame, text="+ Add Assertion", command=self.add_assertion).pack(side="right")

        self.assertions_frame = ctk.CTkScrollableFrame(self.assertions_tab)
        self.assertions_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def add_assertion(self, assertion_data=None):
        if assertion_data is None:
            assertion_data = {"output_variable": "", "assertion_type": "Equals", "expected_value": ""}
            
        frame = ctk.CTkFrame(self.assertions_frame)
        frame.pack(fill="x", pady=2)
        frame.grid_columnconfigure(2, weight=1)
        
        output_vars = self.agent_data.get("outputs", []) + ["status.value"]
        var_menu = ctk.CTkOptionMenu(frame, values=["Select Variable"] + sorted(output_vars))
        var_menu.grid(row=0, column=0, padx=5, pady=5)
        var_menu.set(assertion_data.get("output_variable") or "Select Variable")
        
        assertion_menu = ctk.CTkOptionMenu(frame, values=["Equals", "Regex Match"])
        assertion_menu.grid(row=0, column=1, padx=5, pady=5)
        assertion_menu.set(assertion_data.get("assertion_type", "Equals"))
        
        value_entry = ctk.CTkEntry(frame, placeholder_text="Expected Value")
        value_entry.grid(row=0, column=2, padx=5, pady=5, sticky="ew")
        value_entry.insert(0, assertion_data.get("expected_value", ""))
        
        remove_btn = ctk.CTkButton(frame, text="X", width=30, fg_color=self.get_theme_color('error', 'red'), command=lambda f=frame: f.destroy())
        remove_btn.grid(row=0, column=3, padx=5, pady=5)

    def get_assertions_from_ui(self):
        assertions = []
        for frame in self.assertions_frame.winfo_children():
            if not frame.winfo_exists(): continue
            children = frame.winfo_children()
            var_value = children[0].get()
            if var_value and var_value != "Select Variable":
                assertions.append({
                    "output_variable": var_value,
                    "assertion_type": children[1].get(),
                    "expected_value": children[2].get()
                })
        return assertions
    
    def save_test_case(self):
        name = self.name_entry.get().strip()
        if not name:
            messagebox.showerror("Name Required", "Please enter a name for the test case to save it.")
            return

        current_inputs = {key: var.get() for key, var in self.input_entries.items()}
        current_assertions = self.get_assertions_from_ui()

        new_test_case = {
            "name": name,
            "notes": self.notes_textbox.get("1.0", "end-1c").strip(),
            "date_added": datetime.now().isoformat(),
            "inputs": current_inputs,
            "assertions": current_assertions
        }

        existing_index = -1
        for i, tc in enumerate(self.test_cases):
            if tc.get("name") == name:
                existing_index = i
                break
        
        if existing_index != -1:
            if messagebox.askyesno("Overwrite?", f"A test case named '{name}' already exists. Overwrite it?"):
                self.test_cases[existing_index] = new_test_case
                self.current_test_case_index = existing_index
            else:
                return
        else:
            self.test_cases.append(new_test_case)
            self.current_test_case_index = len(self.test_cases) - 1
            
        self.refresh_test_cases_display()
        self.show_toast(f"Test '{name}' saved.")

    def edit_test_case(self, index):
        self.load_test_case(index)
    
    def delete_test_case(self, index):
        if 0 <= index < len(self.test_cases):
            if messagebox.askyesno("Confirm Delete", f"Delete '{self.test_cases[index]['name']}'?"):
                del self.test_cases[index]
                self.load_scratchpad()
                self.refresh_test_cases_display()
    
    def duplicate_test_case(self, index):
        if 0 <= index < len(self.test_cases):
            original = self.test_cases[index]
            duplicate = copy.deepcopy(original)
            duplicate["name"] = f"{original['name']} (Copy)"
            duplicate["date_added"] = datetime.now().isoformat()
            self.test_cases.append(duplicate)
            self.refresh_test_cases_display()
            
    def refresh_test_cases_display(self):
        for widget in self.test_cases_frame.winfo_children(): widget.destroy()
        if not self.test_cases:
            ctk.CTkLabel(self.test_cases_frame, text="No saved test cases.").pack(pady=20)
            return

        for i, test_case in enumerate(self.test_cases):
            self.create_test_case_widget(i, test_case)

    def create_test_case_widget(self, index, test_case):
        frame = ctk.CTkFrame(self.test_cases_frame)
        frame.pack(fill="x", pady=5, padx=5)
        frame.grid_columnconfigure(0, weight=1)
        
        name = test_case.get("name", f"Test Case {index + 1}")
        note = test_case.get("notes", "")
        
        edit_btn = ctk.CTkButton(frame, text=name, command=lambda i=index: self.edit_test_case(i))
        edit_btn.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        if note: ToolTip(edit_btn, note)
        
        button_frame = ctk.CTkFrame(frame, fg_color="transparent")
        button_frame.grid(row=0, column=1, sticky="e")
        ctk.CTkButton(button_frame, text="Dup", width=40, height=25, command=lambda i=index: self.duplicate_test_case(i)).pack(side="left", padx=2)
        ctk.CTkButton(button_frame, text="Del", width=40, height=25, fg_color=self.get_theme_color('error', 'red'), command=lambda i=index: self.delete_test_case(i)).pack(side="left", padx=2)
    
    def run_all_tests(self):
        if not self.test_cases:
            messagebox.showinfo("No Tests", "There are no saved test cases to run.")
            return
        
        results = []
        for test_case in self.test_cases:
            final_output, final_status, log = self.run_agent_with_inputs(test_case.get("inputs", {}), test_case.get("name"))
            
            assertion_results = []
            case_passed = True
            for assertion in test_case.get("assertions", []):
                passed = self.evaluate_assertion(assertion, final_output, final_status)
                if not passed: case_passed = False
                assertion_results.append({
                    "text": f"{assertion['output_variable']} [{assertion['assertion_type']}] '{assertion['expected_value']}'",
                    "passed": passed
                })
            results.append({
                "name": test_case.get("name"),
                "passed": case_passed,
                "inputs": test_case.get("inputs", {}),
                "assertion_results": assertion_results,
                "final_output": final_output,
                "final_status": final_status,
                "log": log
            })
        
        TestResultsModal(self, results, self.theme)

    def evaluate_assertion(self, assertion, final_result_tape, final_status):
        try:
            output_var = assertion.get("output_variable", "")
            assertion_type = assertion.get("assertion_type", "Equals")
            expected_value = assertion.get("expected_value", "")
            
            if output_var == 'status.value': actual_value = final_status.get('status', {}).get('value')
            else: actual_value = get_nested(final_result_tape, output_var)
            
            if actual_value is None: return False
            
            actual_str, expected_str = str(actual_value), str(expected_value)
            
            if assertion_type == "Equals": return actual_str == expected_str
            if assertion_type == "Regex Match": return bool(re.search(expected_str, actual_str))
            
            return False
        except:
            return False

    def populate_input_fields(self):
        for widget in self.inputs_scroll_frame.winfo_children(): widget.destroy()
        self.input_entries.clear()
        
        required_inputs = self.agent_data.get("inputs", [])
        if required_inputs:
            ctk.CTkLabel(self.inputs_scroll_frame, text="Required", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5)
            for key in required_inputs: self.add_input_widget_for_key(self.inputs_scroll_frame, key, is_optional=False)

        all_optionals = self.agent_data.get("optional_inputs", [])
        if all_optionals:
            ctk.CTkLabel(self.inputs_scroll_frame, text="Optional", font=ctk.CTkFont(weight="bold")).pack(anchor="w", padx=5, pady=(10,0))
            self.optional_menu = ctk.CTkOptionMenu(self.inputs_scroll_frame, values=["Add Optional Input..."] + sorted(all_optionals), command=self.add_optional_from_menu)
            self.optional_menu.pack(anchor="w", padx=5, pady=5)

    def add_input_widget_for_key(self, parent, key, is_optional):
        frame = ctk.CTkFrame(parent, fg_color="transparent")
        frame.pack(fill="x", expand=True, pady=2)

        param_hints = self.agent_data.get("gui", {}).get("param_hints", {})
        hint = param_hints.get(key, {})

        if hint.get("widget_type") == "preset_selector":
            self.create_preset_selector_input_widget(frame, key, hint)
        else:
            self.create_standard_input_widget(frame, key, hint)

        if is_optional:
            remove_btn = ctk.CTkButton(frame, text="X", width=30, fg_color=self.get_theme_color('error', 'red'), command=lambda k=key: self.remove_optional_field(k))
            remove_btn.pack(side="left", padx=5)
            self.optional_frames[key] = frame

    def create_standard_input_widget(self, parent, key, hint):
        ctk.CTkLabel(parent, text=key, width=150).pack(side="left", padx=5)
        entry_var = ctk.StringVar()
        entry = ctk.CTkEntry(parent, textvariable=entry_var)
        entry.pack(side="left", fill="x", expand=True)
        self.input_entries[key] = entry_var
        self.apply_common_param_hints(key, entry, hint)

    def create_preset_selector_input_widget(self, parent, key, hint):
        ctk.CTkLabel(parent, text=key, width=150).pack(side="left", padx=5)
        
        entry_var = ctk.StringVar()
        self.input_entries[key] = entry_var
        
        self.create_preset_selector_widget(parent, key, "", hint, self.app.config_manager)

    def add_optional_from_menu(self, choice):
        if "Add Optional Input..." not in choice and choice not in self.optional_frames:
            self.add_input_widget_for_key(self.inputs_scroll_frame, choice, is_optional=True)
            current_values = self.optional_menu.cget("values")
            current_values.remove(choice)
            self.optional_menu.configure(values=current_values if len(current_values) > 1 else ["No more optionals"])
            self.optional_menu.set("Add Optional Input...")

    def remove_optional_field(self, key):
        if key in self.optional_frames:
            self.optional_frames[key].destroy()
            del self.optional_frames[key]
            del self.input_entries[key]
            if key in self.tooltips:
                self.tooltips[key].destroy()
                del self.tooltips[key]
            
            current_values = self.optional_menu.cget("values")
            if "No more optionals" in current_values: current_values.remove("No more optionals")
            if key not in current_values: current_values.append(key)
            self.optional_menu.configure(values=sorted(current_values))
    
    def load_last_run_config(self):
        run_config = self.agent_data.get("run_config", {})
        
        if "tests" in run_config and "test_cases" not in run_config:
            self.test_cases = [{
                "name": "Imported Legacy Test",
                "notes": "Automatically migrated from the old single-test format.",
                "date_added": datetime.now().isoformat(),
                "inputs": run_config.get("last_inputs", {}),
                "assertions": run_config.get("tests", [])
            }]
        else:
            self.test_cases = run_config.get("test_cases", [])
        
        self.refresh_test_cases_display()
        self.load_scratchpad(run_config)

        self.after(100, self.re_validate_all_inputs)
    
    def load_scratchpad(self, run_config=None):
        if run_config is None: run_config = self.agent_data.get("run_config", {})

        self.current_test_case_index = None
        self.name_entry.delete(0, "end")
        self.notes_textbox.delete("1.0", "end")
        self.name_entry.configure(placeholder_text="<Scratchpad - Unsaved>")
        
        last_inputs = run_config.get("last_inputs", {})
        self.log_level_var.set(run_config.get("last_log_level", "INFO"))

        # Synchronize optional fields to match last_inputs
        current_optionals = set(self.optional_frames.keys())
        last_run_optionals = {k for k in last_inputs if k in self.agent_data.get("optional_inputs", [])}

        for key in current_optionals - last_run_optionals: self.remove_optional_field(key)
        for key in last_run_optionals - current_optionals: self.add_optional_from_menu(key)

        # Now set all values
        for key, var in self.input_entries.items():
            var.set(last_inputs.get(key, ""))
        
        self.refresh_assertions_display([])

    def re_validate_all_inputs(self):
        """
        Forces re-validation on all visible input entry fields and syncs preset selectors.
        """
        param_hints = self.agent_data.get("gui", {}).get("param_hints", {})
            
        # Find all parameter frames currently on screen
        all_param_frames = list(self.optional_frames.values())
        for child in self.inputs_scroll_frame.winfo_children():
            if isinstance(child, ctk.CTkFrame) and hasattr(child, 'winfo_children'):
                if any(isinstance(w, ctk.CTkLabel) for w in child.winfo_children()):
                    all_param_frames.append(child)
        
        for frame in all_param_frames:
            entry_widget = None
            label_widget = None
            
            # Find the label and the main entry widget for this parameter
            for widget in frame.winfo_children():
                if isinstance(widget, ctk.CTkEntry):
                    entry_widget = widget
                elif isinstance(widget, ctk.CTkLabel):
                    label_widget = widget
                # The preset selector has its *own* entry inside an inner frame
                elif isinstance(widget, ctk.CTkFrame):
                    # Find the entry widget inside the preset's frame
                    preset_entry = next((w for w in widget.winfo_children() if isinstance(w, ctk.CTkEntry)), None)
                    if preset_entry:
                        entry_widget = preset_entry

            if entry_widget and label_widget:
                key = label_widget.cget("text")
                hint = param_hints.get(key, {})

                if key in self.input_entries:
                    current_value = self.input_entries[key].get()
                    self.input_entries[key].set(current_value) # This triggers the trace

                # Also run the explicit validation, which was the original purpose
                if "validation" in hint:
                    self.validate_entry_with_feedback(key, entry_widget, hint)

    def load_test_case(self, index):
        if 0 <= index < len(self.test_cases):
            self.current_test_case_index = index
            test_case = self.test_cases[index]
            
            self.name_entry.delete(0, "end")
            self.name_entry.insert(0, test_case.get("name", ""))
            self.notes_textbox.delete("1.0", "end")
            self.notes_textbox.insert("1.0", test_case.get("notes", ""))
            
            inputs_to_load = test_case.get("inputs", {})
            
            current_optionals = set(self.optional_frames.keys())
            test_optionals = {k for k in inputs_to_load if k in self.agent_data.get("optional_inputs", [])}
            
            for key in current_optionals - test_optionals: self.remove_optional_field(key)
            for key in test_optionals - current_optionals: self.add_optional_from_menu(key)
            
            self.after(50, lambda: self.set_all_input_values(inputs_to_load))
            
            self.refresh_assertions_display(test_case.get("assertions", []))
            self.after(100, self.re_validate_all_inputs)

    def set_all_input_values(self, inputs_to_load):
        for key, var in self.input_entries.items():
            var.set(inputs_to_load.get(key, ""))

    def refresh_assertions_display(self, assertions):
        for widget in self.assertions_frame.winfo_children(): widget.destroy()
        for assertion_data in assertions: self.add_assertion(assertion_data)
            
    def save_current_run_config(self):
        # Save the scratchpad state if it's active
        if self.current_test_case_index is None:
             scratchpad_inputs = {key: var.get() for key, var in self.input_entries.items()}
        else:
             # Otherwise, preserve the last known scratchpad state
             scratchpad_inputs = self.agent_data.get("run_config", {}).get("last_inputs", {})

        clean_test_cases = []
        for tc in self.test_cases:
            clean_case = tc.copy()
            clean_case.pop('_result_label', None)
            clean_test_cases.append(clean_case)
            
        new_run_config = {
            "last_inputs": scratchpad_inputs,
            "last_log_level": self.log_level_var.get(), 
            "test_cases": clean_test_cases
        }
        
        if self.app.editor_frame_instance and hasattr(self.app.editor_frame_instance, 'data'):
            self.app.editor_frame_instance.data['run_config'] = new_run_config
