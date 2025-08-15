# --- START OF FILE ui_editors.py ---
import customtkinter as ctk
import json
import copy
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token
from tkinter import messagebox

class CTkCodeEditor(ctk.CTkFrame):
    def __init__(self, master, theme, language="python", **kwargs):
        super().__init__(master, **kwargs)
        self.theme = theme
        self.language_lexer = PythonLexer()
        self.grid_columnconfigure(1, weight=1); self.grid_rowconfigure(0, weight=1)
        
        code_font = ctk.CTkFont(family=theme['fonts']['editor_code_family'], size=theme['fonts']['editor_code_size'])

        four_digit_width = code_font.measure("9999") + 20 
        self.line_numbers = ctk.CTkTextbox(self, width=four_digit_width, font=code_font, 
                                           fg_color=theme['colors']['bg_tertiary'], 
                                           text_color=theme['colors']['text_secondary'])

        self.line_numbers.grid(row=0, column=0, sticky="ns"); self.line_numbers.insert("1.0", "1"); self.line_numbers.configure(state="disabled")
        
        self.textbox = ctk.CTkTextbox(self, font=code_font, wrap="none", 
                                      fg_color=theme['colors']['editor_code_bg'], 
                                      text_color=theme['colors']['editor_code_text'])
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

class ListEditorFrame(ctk.CTkFrame):
    def __init__(self, master, title, initial_list, theme, sync_callback=None):
        super().__init__(master, fg_color="transparent")
        self.items = list(initial_list)
        self.theme = theme
        self.sync_callback = sync_callback  # Callback for auto-sync

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'], weight="bold")
        ctk.CTkLabel(self, text=title, font=label_font, text_color=self.theme['colors']['text_primary']).grid(row=0, column=0, sticky="w", padx=5, pady=2)
        
        self.entries_frame = ctk.CTkScrollableFrame(self, fg_color=self.theme['colors']['bg_tertiary'])
        self.entries_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        ctk.CTkButton(self, text="+ Add", width=80, command=self.add_item).grid(row=2, column=0, pady=(0, 5), padx=5)
        
        self.refresh()

    def add_item(self, value=""): 
        self.items.append(value)
        self.refresh()
        if self.sync_callback:
            self.sync_callback()
    
    def remove_item(self, index): 
        self.items.pop(index)
        self.refresh()
        if self.sync_callback:
            self.sync_callback()
    
    def get_data(self): 
        return [widget.get() for row in self.entries_frame.winfo_children() if isinstance(row, ctk.CTkFrame) for widget in row.winfo_children() if isinstance(widget, ctk.CTkEntry)]

    def refresh(self):
        for widget in self.entries_frame.winfo_children():
            widget.destroy()
            
        entry_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['main_size'])
        for i, item in enumerate(self.items):
            row_frame = ctk.CTkFrame(self.entries_frame, fg_color="transparent")
            row_frame.pack(fill="x", pady=2, padx=2)
            entry = ctk.CTkEntry(row_frame, font=entry_font)
            entry.insert(0, item)
            entry.pack(side="left", fill="x", expand=True)
            # Add sync on text change
            entry.bind("<KeyRelease>", lambda e: self.sync_callback() if self.sync_callback else None)
            remove_btn = ctk.CTkButton(row_frame, text="X", width=30, fg_color=self.theme['colors']['error'], command=lambda index=i: self.remove_item(index))
            remove_btn.pack(side="left", padx=5)

class BaseEditorFrame(ctk.CTkFrame):
    def __init__(self, master, agent_name, agent_data, app_ref, theme):
        super().__init__(master, fg_color=theme['colors']['bg_secondary'])
        self.agent_name = agent_name
        self.data = agent_data
        self.app_ref = app_ref
        self.theme = theme
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

    def get_data(self): raise NotImplementedError

    def copy_agent_definition_to_clipboard(self):
        """Gathers the current agent data, formats it as JSON, and copies it."""
        if not hasattr(self, 'get_data'):
            print("Error: The 'get_data' method is not implemented in this editor.")
            return
            
        # 1. Get the most up-to-date data from the form fields
        current_data = self.get_data()
        if current_data is None:
            # This can happen if there's a validation error (e.g., in the JSON editor)
            from tkinter import messagebox
            messagebox.showerror("Cannot Copy", "Could not retrieve agent data. Please check the editor for errors (e.g., invalid JSON).")
            return
        
        # 2. Get the current name and format the final JSON object
        agent_name = current_data.pop('name', self.agent_name)
        
        # The final structure to be copied is {"agent_name": { ...agent_data... }}
        json_to_copy = {agent_name: current_data}
        
        # 3. Format as a pretty string and copy to clipboard
        try:
            json_string = json.dumps(json_to_copy, indent=2)
            
            self.clipboard_clear()
            self.clipboard_append(json_string)
            
            # Use the app_ref to show a toast message for feedback
            self.app_ref.show_toast(f"Definition for '{agent_name}' copied to clipboard.")
            
        except Exception as e:
            from tkinter import messagebox
            messagebox.showerror("Copy Failed", f"An unexpected error occurred while preparing the JSON data: {e}")

    def _create_help_button(self, parent, help_text):
        def show_help(): self.app_ref.show_help_modal("Help", help_text)
        return ctk.CTkButton(parent, text="?", width=25, height=25, command=show_help)

    def _get_io_help_text(self):
        return """
Inputs, Outputs, and Optional Inputs define the 'signature' of your agent.
- Inputs: Required parameters.
- Optional Inputs: Non-required parameters.
- Outputs: Variables the agent will produce.
"""

    def setup_auto_sync(self):
        """Setup automatic synchronization of UI changes to underlying data"""
        # Common fields that exist in both proc and template editors
        if hasattr(self, 'name_entry'):
            self.name_entry.bind("<KeyRelease>", self.sync_name)
        if hasattr(self, 'help_text'):
            self.help_text.bind("<KeyRelease>", self.sync_help)
        if hasattr(self, 'web_services_entry'):
            self.web_services_entry.bind("<KeyRelease>", self.sync_web_services)
        
        # ProcEditorFrame specific
        if hasattr(self, 'func_def_text'):
            self.func_def_text.textbox.bind("<KeyRelease>", self.sync_function_def)
        if hasattr(self, 'func_name_entry'):
            self.func_name_entry.bind("<KeyRelease>", self.sync_function_name)
        
        # TemplateEditorFrame specific  
        if hasattr(self, 'prompt_text'):
            self.prompt_text.bind("<KeyRelease>", self.sync_prompt)

    def sync_name(self, event=None):
        """Sync agent name to data"""
        if hasattr(self, 'name_entry'):
            new_name = self.name_entry.get().strip()
            if new_name != self.agent_name:
                self.agent_name = new_name
                self.data['name'] = new_name

    def sync_help(self, event=None):
        """Sync help text to data"""
        if hasattr(self, 'help_text'):
            self.data['help'] = self.help_text.get("1.0", "end-1c").strip()

    def sync_web_services(self, event=None):
        """Sync web services to data"""
        if hasattr(self, 'web_services_entry'):
            tags_string = self.web_services_entry.get().strip()
            tags_list = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
            if tags_list:
                self.data['web_services'] = tags_list
            elif 'web_services' in self.data:
                del self.data['web_services']

    def sync_function_def(self, event=None):
        """Sync function definition text to data"""
        if hasattr(self, 'func_def_text'):
            self.data['function_def'] = self.func_def_text.get("1.0", "end-1c").strip()

    def sync_function_name(self, event=None):
        """Sync function name to data"""
        if hasattr(self, 'func_name_entry'):
            self.data['function'] = self.func_name_entry.get()

    def sync_prompt(self, event=None):
        """Sync prompt text to data"""
        if hasattr(self, 'prompt_text'):
            self.data['prompt'] = self.prompt_text.get("1.0", "end-1c").strip()

    def sync_io_data(self):
        """Sync input/output lists to data"""
        if hasattr(self, 'inputs_frame'):
            self.data['inputs'] = self.inputs_frame.get_data()
        if hasattr(self, 'optionals_frame'):
            self.data['optional_inputs'] = self.optionals_frame.get_data()
        if hasattr(self, 'outputs_frame'):
            self.data['outputs'] = self.outputs_frame.get_data()

class ValidationRulesModal(ctk.CTkToplevel):
    def __init__(self, parent, validation_data=None):
        super().__init__(parent)
        self.master = parent # Store parent reference
        self.title("Edit Validation Rules")
        self.geometry("400x350")
        self.validation_data = copy.deepcopy(validation_data) or {}
        self.saved = False
        self.create_widgets()
        self.transient(parent)
        self.grab_set()

    def create_widgets(self):
        self.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(self, text="Type:").grid(row=0, column=0, padx=10, pady=5, sticky="w")
        self.type_var = ctk.StringVar(value=self.validation_data.get("type", "None"))
        type_menu = ctk.CTkOptionMenu(self, variable=self.type_var, values=["None", "string", "integer", "float"])
        type_menu.grid(row=0, column=1, padx=10, pady=5, sticky="ew")

        ctk.CTkLabel(self, text="Min Value:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.min_entry = ctk.CTkEntry(self)
        self.min_entry.grid(row=1, column=1, padx=10, pady=5, sticky="ew")
        if "min_value" in self.validation_data: self.min_entry.insert(0, str(self.validation_data["min_value"]))

        ctk.CTkLabel(self, text="Max Value:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.max_entry = ctk.CTkEntry(self)
        self.max_entry.grid(row=2, column=1, padx=10, pady=5, sticky="ew")
        if "max_value" in self.validation_data: self.max_entry.insert(0, str(self.validation_data["max_value"]))
        
        ctk.CTkLabel(self, text="Regex:").grid(row=3, column=0, padx=10, pady=5, sticky="w")
        
        gui_patterns = self.master.app_ref.config_manager.config.get("GUI", {}).get("regex_patterns", {})
        pattern_options = ["Custom..."] + sorted(gui_patterns.keys())
        
        regex_var = ctk.StringVar()
        regex_menu = ctk.CTkOptionMenu(self, variable=regex_var, values=pattern_options)
        regex_menu.grid(row=3, column=1, sticky="ew", padx=10, pady=5)
        
        self.regex_entry = ctk.CTkEntry(self)
        self.regex_entry.grid(row=4, column=1, sticky="ew", padx=10, pady=5)
        if "regex" in self.validation_data: self.regex_entry.insert(0, self.validation_data["regex"])
        
        def on_regex_menu_select(choice):
            if choice != "Custom...":
                pattern = gui_patterns.get(choice, "")
                self.regex_entry.delete(0, "end")
                self.regex_entry.insert(0, pattern)

        def on_regex_entry_change(*args):
            regex_var.set("Custom...")

        regex_menu.configure(command=on_regex_menu_select)
        self.regex_entry.bind("<KeyRelease>", on_regex_entry_change)

        initial_regex = self.validation_data.get("regex", "")
        matching_key = "Custom..."
        for key, value in gui_patterns.items():
            if value == initial_regex:
                matching_key = key
                break
        regex_var.set(matching_key)

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        ctk.CTkButton(button_frame, text="Cancel", command=self.destroy).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save", command=self.save).pack(side="right", padx=10)

    def save(self):
        selected_type = self.type_var.get()
        if selected_type == "None":
            self.validation_data = {}
            self.saved = True
            self.destroy()
            return

        new_data = {"type": selected_type}
        try:
            if selected_type in ["integer", "float"]:
                if self.min_entry.get(): new_data["min_value"] = float(self.min_entry.get()) if selected_type == "float" else int(self.min_entry.get())
                if self.max_entry.get(): new_data["max_value"] = float(self.max_entry.get()) if selected_type == "float" else int(self.max_entry.get())
            
            if selected_type == "string":
                if self.regex_entry.get():
                    new_data["regex"] = self.regex_entry.get()
                                      
            self.validation_data = new_data
            self.saved = True
            self.destroy()
        except ValueError: messagebox.showerror("Error", "Min/Max values must be valid numbers.", parent=self)

class GuiHintsEditorFrame(ctk.CTkFrame):
    def __init__(self, master, agent_data, theme, app_ref, sync_callback=None):
        super().__init__(master, fg_color="transparent")
        self.agent_data = agent_data
        self.theme = theme
        self.app_ref = app_ref
        self.hint_cards = {}
        self.sync_callback = sync_callback  # Callback for auto-sync
        
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        control_frame = ctk.CTkFrame(self, fg_color="transparent")
        control_frame.grid(row=0, column=0, sticky="ew", padx=5, pady=5)
        self.add_hint_menu = ctk.CTkOptionMenu(control_frame, command=self.add_hint_card)
        self.add_hint_menu.pack(side="left")

        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.load_existing_hints()
        self.populate_dropdown()

    def populate_dropdown(self):
        # This function now correctly uses the stale self.agent_data, but filters
        # against the live self.hint_cards dictionary, which is the correct logic.
        all_params = self.agent_data.get("inputs", []) + self.agent_data.get("optional_inputs", [])
        
        # The key is to check what's currently a card.
        active_params = self.hint_cards.keys() 
        
        available_params = sorted([p for p in all_params if p not in active_params and p])
        
        if not available_params:
            self.add_hint_menu.configure(values=["No available parameters"], state="disabled")
            self.add_hint_menu.set("No available parameters")
        else:
            self.add_hint_menu.configure(values=["Add Hint for Parameter..."] + available_params, state="normal")
            self.add_hint_menu.set("Add Hint for Parameter...")

    def load_existing_hints(self):
        param_hints = self.agent_data.get("gui", {}).get("param_hints", {})
        for param_name, hint_data in param_hints.items():
            self._create_hint_card(param_name, hint_data)

    def add_hint_card(self, param_name):
        if "Add Hint" in param_name: return
        self._create_hint_card(param_name)
        self.populate_dropdown()
        if self.sync_callback:
            self.sync_callback()

    def _create_hint_card(self, param_name, hint_data=None):
        if hint_data is None: hint_data = {}
        
        card = ctk.CTkFrame(self.scroll_frame, border_width=1)
        card.pack(fill="x", padx=5, pady=5)
        card.grid_columnconfigure(1, weight=1)
        
        header = ctk.CTkFrame(card, fg_color="transparent")
        header.grid(row=0, column=0, columnspan=2, sticky="ew", padx=5, pady=5)
        ctk.CTkLabel(header, text=f"Hint for: {param_name}", font=ctk.CTkFont(weight="bold")).pack(side="left")
        remove_btn = ctk.CTkButton(header, text="X", width=30, fg_color="red", command=lambda p=param_name: self.remove_hint_card(p))
        remove_btn.pack(side="right")
        
        widgets = {}
        ctk.CTkLabel(card, text="Tooltip:").grid(row=1, column=0, sticky="w", padx=10)
        tooltip_entry = ctk.CTkEntry(card)
        tooltip_entry.grid(row=1, column=1, sticky="ew", padx=5, pady=2)
        tooltip_entry.insert(0, hint_data.get("tooltip", ""))
        tooltip_entry.bind("<KeyRelease>", lambda e: self.sync_callback() if self.sync_callback else None)
        widgets["tooltip"] = tooltip_entry

        ctk.CTkLabel(card, text="Example:").grid(row=2, column=0, sticky="w", padx=10)
        example_entry = ctk.CTkEntry(card)
        example_entry.grid(row=2, column=1, sticky="ew", padx=5, pady=2)
        example_entry.insert(0, hint_data.get("example", ""))
        example_entry.bind("<KeyRelease>", lambda e: self.sync_callback() if self.sync_callback else None)
        widgets["example"] = example_entry

        ctk.CTkLabel(card, text="Data Source:").grid(row=3, column=0, sticky="w", padx=10)
        
        gui_sources = self.app_ref.config_manager.config.get("GUI", {})
        data_source_options = ["Default (Custom)"] + sorted([f"GUI.{key}" for key in gui_sources.keys()])
        
        data_source_entry = ctk.CTkEntry(card)
        data_source_entry.grid(row=4, column=1, sticky="ew", padx=5, pady=2)
        data_source_entry.insert(0, hint_data.get("data_source", ""))
        data_source_entry.bind("<KeyRelease>", lambda e: self.sync_callback() if self.sync_callback else None)
        widgets["data_source"] = data_source_entry

        data_source_var = ctk.StringVar()
        data_source_menu = ctk.CTkOptionMenu(card, variable=data_source_var, values=data_source_options)
        data_source_menu.grid(row=3, column=1, sticky="ew", padx=5, pady=2)
        
        validation_frame = ctk.CTkFrame(card, fg_color="transparent")
        validation_frame.grid(row=5, column=1, sticky="w", padx=5, pady=2)
        validation_btn = ctk.CTkButton(validation_frame, text="Edit Validation...")
        widgets["validation_btn"] = validation_btn
        validation_btn.pack(side="left")
        widgets["validation_data"] = hint_data.get("validation", {})
        
        def on_menu_select(choice):
            if choice != "Default (Custom)":
                data_source_entry.delete(0, "end")
                data_source_entry.insert(0, choice)
                if self.sync_callback:
                    self.sync_callback()

        def on_entry_change(*args):
            data_source_var.set("Default (Custom)")

        data_source_menu.configure(command=on_menu_select)
        data_source_entry.bind("<KeyRelease>", on_entry_change)

        initial_ds = hint_data.get("data_source", "")
        if initial_ds in data_source_options:
            data_source_var.set(initial_ds)
        else:
            data_source_var.set("Default (Custom)")
            
        def update_validation_button_state(*args):
            if tooltip_entry.get(): validation_btn.configure(state="normal")
            else: validation_btn.configure(state="disabled")
        tooltip_entry.bind("<KeyRelease>", update_validation_button_state)
        update_validation_button_state()
        
        def open_validation_modal():
            modal = ValidationRulesModal(self, widgets["validation_data"])
            self.wait_window(modal)
            if modal.saved: 
                widgets["validation_data"] = modal.validation_data
                if self.sync_callback:
                    self.sync_callback()
        validation_btn.configure(command=open_validation_modal)
        
        self.hint_cards[param_name] = {"card": card, "widgets": widgets}

    def remove_hint_card(self, param_name):
        if param_name in self.hint_cards:
            self.hint_cards[param_name]["card"].destroy()
            del self.hint_cards[param_name]
            self.populate_dropdown()
            if self.sync_callback:
                self.sync_callback()

    def get_data(self):
        param_hints = {}
        for param_name, card_info in self.hint_cards.items():
            widgets = card_info["widgets"]
            current_hint = {}
            if tooltip := widgets["tooltip"].get(): current_hint["tooltip"] = tooltip
            if example := widgets["example"].get(): current_hint["example"] = example
            if data_source := widgets["data_source"].get():
                current_hint["widget_type"] = "preset_selector"
                current_hint["data_source"] = data_source
            if current_hint.get("tooltip") and widgets["validation_data"]:
                current_hint["validation"] = widgets["validation_data"]
            if current_hint:
                param_hints[param_name] = current_hint
        return param_hints

class GuiSettingsModal(ctk.CTkToplevel):
    def __init__(self, parent, gui_data, theme):
        super().__init__(parent)
        self.title("Advanced GUI Settings")
        self.geometry("800x700")
        self.gui_data = copy.deepcopy(gui_data) or {}
        self.theme = theme
        self.saved = False
        self.script_widgets = []
        self.app_ref = parent.app_ref
        self.create_widgets()
        self.transient(parent)
        self.grab_set()
        
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
        code_editor = CTkCodeEditor(frame, height=150, theme=self.theme); code_editor.insert("1.0", code)
        code_editor.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="nsew")
        widget_set = {'frame': frame, 'name': name_entry, 'code': code_editor, 'btn': remove_btn}
        remove_btn.configure(command=lambda w=widget_set: self.remove_script_action(w))
        self.script_widgets.append(widget_set)

    def remove_script_action(self, widget_set): widget_set['frame'].destroy(); self.script_widgets.remove(widget_set)
    def show_help(self): self.app_ref.show_help_modal("GUI Settings Help", """...""")
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

class ProcEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref, theme):
        super().__init__(master, agent_name, agent_data, app_ref, theme)
        
        self.GuiSettingsModal = GuiSettingsModal
        
        tab_view = ctk.CTkTabview(self, fg_color=self.theme['colors']['bg_primary'])
        tab_view.grid(row=0, column=0, sticky="nsew")
        
        self.create_settings_tab(tab_view.add("Settings"))
        self.create_gui_hints_tab(tab_view.add("GUI Hints"))
        self.create_inputs_tab(tab_view.add("Inputs"))
        self.create_optionals_tab(tab_view.add("Optional Inputs"))
        self.create_outputs_tab(tab_view.add("Outputs"))
        self.create_function_tab(tab_view.add("Function"))
        
        # Setup auto-sync after all UI elements are created
        self.setup_auto_sync()

    def create_settings_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'], weight="bold")
        main_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['main_size'])
        
        ctk.CTkLabel(tab, text="Agent Name:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.name_entry = ctk.CTkEntry(tab, font=main_font)
        self.name_entry.insert(0, self.agent_name)
        self.name_entry.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(tab, text="Help Text:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.help_text = ctk.CTkTextbox(tab, height=200, font=main_font)
        self.help_text.insert("1.0", self.data.get("help", ""))
        self.help_text.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(tab, text="Web Service Tags (comma-separated):", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.web_services_entry = ctk.CTkEntry(tab, font=main_font)
        self.web_services_entry.insert(0, ", ".join(self.data.get("web_services", [])))
        self.web_services_entry.pack(fill="x", padx=10, pady=5)

        copy_button = ctk.CTkButton(tab, text="Copy Agent Definition to Clipboard", command=self.copy_agent_definition_to_clipboard)
        copy_button.pack(fill="x", padx=10, pady=(15, 5))

    def create_gui_hints_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        tab.grid_rowconfigure(0, weight=1); tab.grid_columnconfigure(0, weight=1)
        self.gui_hints_frame = GuiHintsEditorFrame(tab, self.data, self.theme, self.app_ref, self.sync_gui_hints)
        self.gui_hints_frame.grid(row=0, column=0, sticky="nsew")
        ctk.CTkButton(tab, text="Advanced GUI Settings (for Workflow Editor)...", command=self.open_gui_settings).grid(row=1, column=0, sticky="ew", padx=10, pady=10)

    def create_inputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.inputs_frame = ListEditorFrame(tab, "Required Inputs", self.data.get("inputs", []), self.theme, self.sync_io_data); self.inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_optionals_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.optionals_frame = ListEditorFrame(tab, "Optional Inputs", self.data.get("optional_inputs", []), self.theme, self.sync_io_data); self.optionals_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_outputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.outputs_frame = ListEditorFrame(tab, "Outputs", self.data.get("outputs", []), self.theme, self.sync_io_data); self.outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_function_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        tab.grid_columnconfigure(0, weight=1); tab.grid_rowconfigure(3, weight=1)
        
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'])
        main_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['main_size'])

        ctk.CTkLabel(tab, text="Function Name", font=label_font).grid(row=0, column=0, sticky="w", padx=5, pady=(10,0))
        self.func_name_entry = ctk.CTkEntry(tab, font=main_font); self.func_name_entry.insert(0, self.data.get("function", "")); self.func_name_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        ctk.CTkLabel(tab, text="Function Definition", font=label_font).grid(row=2, column=0, sticky="w", padx=5, pady=(10,0))
        self.func_def_text = CTkCodeEditor(tab, theme=self.theme); self.func_def_text.insert("1.0", self.data.get("function_def", "")); self.func_def_text.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    
    def sync_gui_hints(self):
        """Sync GUI hints to data"""
        param_hints = self.gui_hints_frame.get_data()
        other_gui_data = {k: v for k, v in self.data.get("gui", {}).items() if k != 'param_hints'}
        if param_hints or other_gui_data:
            self.data['gui'] = other_gui_data
            if param_hints:
                self.data['gui']['param_hints'] = param_hints
        elif 'gui' in self.data:
            del self.data['gui']

    def open_gui_settings(self):
        # This modal only edits the non-param_hints parts of the gui key
        other_gui_data = {k: v for k, v in self.data.get("gui", {}).items() if k != 'param_hints'}
        modal = self.GuiSettingsModal(self, other_gui_data, self.theme)
        self.wait_window(modal)
        if modal.saved:
            updated_other_gui_data = modal.get_result()
            self.data.setdefault("gui", {})
            # Clear old non-param_hints keys
            for key in list(self.data["gui"].keys()):
                if key != 'param_hints': del self.data["gui"][key]
            # Add new non-param_hints keys
            if updated_other_gui_data:
                self.data["gui"].update(updated_other_gui_data)

    def get_data(self):
        updated_data = copy.deepcopy(self.data)
        updated_data['name'] = self.name_entry.get().strip()
        updated_data['help'] = self.help_text.get("1.0", "end-1c").strip()
        
        tags_string = self.web_services_entry.get().strip()
        tags_list = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
        if tags_list: updated_data['web_services'] = tags_list
        elif 'web_services' in updated_data: del updated_data['web_services']
            
        updated_data['inputs'] = self.inputs_frame.get_data()
        updated_data['optional_inputs'] = self.optionals_frame.get_data()
        updated_data['outputs'] = self.outputs_frame.get_data()
        updated_data['function'] = self.func_name_entry.get()
        updated_data['function_def'] = self.func_def_text.get("1.0", "end-1c").strip()
        
        # Handle GUI data
        param_hints = self.gui_hints_frame.get_data()
        other_gui_data = {k: v for k, v in updated_data.get("gui", {}).items() if k != 'param_hints'}
        if param_hints or other_gui_data:
            updated_data['gui'] = other_gui_data
            if param_hints:
                updated_data['gui']['param_hints'] = param_hints
        elif 'gui' in updated_data:
            del updated_data['gui']
            
        return updated_data

class TemplateEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref, theme):
        super().__init__(master, agent_name, agent_data, app_ref, theme)
        tab_view = ctk.CTkTabview(self, fg_color=self.theme['colors']['bg_primary'])
        tab_view.grid(row=0, column=0, sticky="nsew")

        self.create_settings_tab(tab_view.add("Settings"))
        self.create_gui_hints_tab(tab_view.add("GUI Hints"))
        self.create_inputs_tab(tab_view.add("Inputs"))
        self.create_optionals_tab(tab_view.add("Optional Inputs"))
        self.create_outputs_tab(tab_view.add("Outputs"))
        self.create_prompt_tab(tab_view.add("Prompt"))
        
        # Setup auto-sync after all UI elements are created
        self.setup_auto_sync()

    def create_settings_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'], weight="bold")
        main_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['main_size'])
        
        ctk.CTkLabel(tab, text="Agent Name:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.name_entry = ctk.CTkEntry(tab, font=main_font)
        self.name_entry.insert(0, self.agent_name)
        self.name_entry.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(tab, text="Help Text:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.help_text = ctk.CTkTextbox(tab, height=200, font=main_font)
        self.help_text.insert("1.0", self.data.get("help", ""))
        self.help_text.pack(fill="x", padx=10, pady=5)

        ctk.CTkLabel(tab, text="Web Service Tags (comma-separated):", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.web_services_entry = ctk.CTkEntry(tab, font=main_font)
        self.web_services_entry.insert(0, ", ".join(self.data.get("web_services", [])))
        self.web_services_entry.pack(fill="x", padx=10, pady=5)

        copy_button = ctk.CTkButton(tab, text="Copy Agent Definition to Clipboard", command=self.copy_agent_definition_to_clipboard)
        copy_button.pack(fill="x", padx=10, pady=(15, 5))

    def create_gui_hints_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        tab.grid_rowconfigure(0, weight=1); tab.grid_columnconfigure(0, weight=1)
        self.gui_hints_frame = GuiHintsEditorFrame(tab, self.data, self.theme, self.app_ref, self.sync_gui_hints)
        self.gui_hints_frame.grid(row=0, column=0, sticky="nsew")

    def create_inputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.inputs_frame = ListEditorFrame(tab, "Required Inputs", self.data.get("inputs", []), self.theme, self.sync_io_data); self.inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_optionals_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.optionals_frame = ListEditorFrame(tab, "Optional Inputs", self.data.get("optional_inputs", []), self.theme, self.sync_io_data); self.optionals_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_outputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.outputs_frame = ListEditorFrame(tab, "Outputs", self.data.get("outputs", []), self.theme, self.sync_io_data); self.outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_prompt_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        tab.grid_rowconfigure(1, weight=1); tab.grid_columnconfigure(0, weight=1)
        
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'])
        template_font = ctk.CTkFont(family=self.theme['fonts']['editor_template_family'], size=self.theme['fonts']['editor_template_size'])

        ctk.CTkLabel(tab, text="Prompt Template", font=label_font).grid(row=0, column=0, pady=(5,0))
        self.prompt_text = ctk.CTkTextbox(tab, font=template_font, 
                                          fg_color=self.theme['colors']['editor_template_bg'], 
                                          text_color=self.theme['colors']['editor_template_text'])
        self.prompt_text.insert("1.0", self.data.get("prompt", ""))
        self.prompt_text.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)

    def sync_gui_hints(self):
        """Sync GUI hints to data"""
        param_hints = self.gui_hints_frame.get_data()
        if param_hints:
            self.data.setdefault('gui', {})['param_hints'] = param_hints
        elif 'gui' in self.data and 'param_hints' in self.data['gui']:
            del self.data['gui']['param_hints']
            if not self.data['gui']:
                del self.data['gui']

    def get_data(self):
        updated_data = copy.deepcopy(self.data)
        updated_data['name'] = self.name_entry.get().strip()
        updated_data['help'] = self.help_text.get("1.0", "end-1c").strip()
        
        tags_string = self.web_services_entry.get().strip()
        tags_list = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
        if tags_list: updated_data['web_services'] = tags_list
        elif 'web_services' in updated_data: del updated_data['web_services']
            
        updated_data['inputs'] = self.inputs_frame.get_data()
        updated_data['optional_inputs'] = self.optionals_frame.get_data()
        updated_data['outputs'] = self.outputs_frame.get_data()
        updated_data['prompt'] = self.prompt_text.get("1.0", "end-1c").strip()

        # Handle GUI data
        param_hints = self.gui_hints_frame.get_data()
        if param_hints:
            updated_data.setdefault('gui', {})['param_hints'] = param_hints
        elif 'gui' in updated_data and 'param_hints' in updated_data['gui']:
            del updated_data['gui']['param_hints']
            if not updated_data['gui']:
                del updated_data['gui']

        return updated_data

class JsonEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref, theme):
        super().__init__(master, agent_name, agent_data, app_ref, theme)
        
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=16, weight="bold")
        code_font = ctk.CTkFont(family=self.theme['fonts']['code_family'], size=self.theme['fonts']['code_size'])

        ctk.CTkLabel(self, text="Raw JSON Editor", font=label_font).pack(anchor="w", padx=10, pady=(10,0))
        self.textbox = ctk.CTkTextbox(self, font=code_font); self.textbox.pack(fill="both", expand=True, padx=10, pady=10)
        self.textbox.insert("1.0", json.dumps(self.data, indent=2))
        
        # Setup auto-sync for JSON editor
        self.textbox.bind("<KeyRelease>", self.sync_json_data)
        
    def sync_json_data(self, event=None):
        """Sync JSON text to data - but handle errors gracefully"""
        try:
            json_text = self.textbox.get("1.0", "end-1c")
            parsed_data = json.loads(json_text)
            # Only update if JSON is valid
            if isinstance(parsed_data, dict):
                self.data.clear()
                self.data.update(parsed_data)
        except json.JSONDecodeError:
            # Invalid JSON - don't update data, just continue
            pass
        
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
