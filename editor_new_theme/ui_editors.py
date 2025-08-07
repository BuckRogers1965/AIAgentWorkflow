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

        self.line_numbers = ctk.CTkTextbox(self, width=40, font=code_font, 
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
    def __init__(self, master, title, initial_list, theme):
        super().__init__(master, fg_color="transparent")
        self.items = list(initial_list)
        self.theme = theme

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
    
    def remove_item(self, index): 
        self.items.pop(index)
        self.refresh()
    
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
        self.create_inputs_tab(tab_view.add("Inputs"))
        self.create_optionals_tab(tab_view.add("Optional Inputs"))
        self.create_outputs_tab(tab_view.add("Outputs"))
        self.create_function_tab(tab_view.add("Function"))

    def create_settings_tab(self, tab):
        #print("\n" + "="*20 + f" INSTRUMENTATION START: ProcEditorFrame for '{self.agent_name}' " + "="*20)
        #print(f"STEP 0: Full self.data dictionary received by editor:\n{json.dumps(self.data, indent=2)}")

        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        #print("STEP 1: Tab configured.")

        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'], weight="bold")
        main_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['main_size'])
        #print("STEP 2: Fonts created.")
        
        #print("STEP 3: Creating Agent Name widgets.")
        ctk.CTkLabel(tab, text="Agent Name:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.name_entry = ctk.CTkEntry(tab, font=main_font)
        #print(f"STEP 3a: Inserting agent name '{self.agent_name}' into name_entry.")
        self.name_entry.insert(0, self.agent_name)
        self.name_entry.pack(fill="x", padx=10, pady=5)
        #print("STEP 3b: Agent Name widgets created and populated.")
        
        #print("STEP 4: Creating Help Text widgets.")
        ctk.CTkLabel(tab, text="Help Text:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.help_text = ctk.CTkTextbox(tab, height=200, font=main_font)
        help_data = self.data.get("help", "")
        #print(f"STEP 4a: Inserting help text '{help_data[:50]}...' into help_text.")
        self.help_text.insert("1.0", help_data)
        self.help_text.pack(fill="x", padx=10, pady=5)
        #print("STEP 4b: Help Text widgets created and populated.")

        #print("STEP 5: Creating Web Service Tags widgets.")
        ctk.CTkLabel(tab, text="Web Service Tags (comma-separated):", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.web_services_entry = ctk.CTkEntry(tab, font=main_font)
        tags_list = self.data.get("web_services", [])
        #print(f"STEP 5a: Got web_services list from self.data: {tags_list} (Type: {type(tags_list)})")
        display_string = ", ".join(tags_list)
        #print(f"STEP 5b: Converted list to string: '{display_string}'")
        #print(f"STEP 5c: Inserting string into web_services_entry.")
        self.web_services_entry.insert(0, display_string)
        self.web_services_entry.pack(fill="x", padx=10, pady=5)
        #print("STEP 5d: Web Service Tags widgets created and populated.")
        
        #print("STEP 6: Creating Advanced GUI Settings button.")
        ctk.CTkButton(tab, text="Advanced GUI Settings...", command=self.open_gui_settings).pack(anchor="w", padx=10, pady=10)
        #print("="*20 + " INSTRUMENTATION END: ProcEditorFrame " + "="*20 + "\n")


    def create_inputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.inputs_frame = ListEditorFrame(tab, "Required Inputs", self.data.get("inputs", []), self.theme); self.inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_optionals_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.optionals_frame = ListEditorFrame(tab, "Optional Inputs", self.data.get("optional_inputs", []), self.theme); self.optionals_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_outputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.outputs_frame = ListEditorFrame(tab, "Outputs", self.data.get("outputs", []), self.theme); self.outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_function_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        tab.grid_columnconfigure(0, weight=1); tab.grid_rowconfigure(3, weight=1)
        
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'])
        main_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['main_size'])

        ctk.CTkLabel(tab, text="Function Name", font=label_font).grid(row=0, column=0, sticky="w", padx=5, pady=(10,0))
        self.func_name_entry = ctk.CTkEntry(tab, font=main_font); self.func_name_entry.insert(0, self.data.get("function", "")); self.func_name_entry.grid(row=1, column=0, sticky="ew", padx=5, pady=5)
        
        ctk.CTkLabel(tab, text="Function Definition", font=label_font).grid(row=2, column=0, sticky="w", padx=5, pady=(10,0))
        self.func_def_text = CTkCodeEditor(tab, theme=self.theme); self.func_def_text.insert("1.0", self.data.get("function_def", "")); self.func_def_text.grid(row=3, column=0, sticky="nsew", padx=5, pady=5)
    
    def open_gui_settings(self):
        modal = self.GuiSettingsModal(self, self.data.get("gui", {}), self.theme); self.wait_window(modal)
        if modal.saved:
            updated_gui_data = modal.get_result()
            if updated_gui_data: self.data["gui"] = updated_gui_data
            elif "gui" in self.data: del self.data["gui"]

    def get_data(self):
        updated_data = copy.deepcopy(self.data)
        updated_data['name'] = self.name_entry.get().strip()
        updated_data['help'] = self.help_text.get("1.0", "end-1c").strip()
        
        tags_string = self.web_services_entry.get().strip()
        tags_list = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
        if tags_list:
            updated_data['web_services'] = tags_list
        elif 'web_services' in updated_data:
            del updated_data['web_services']
            
        updated_data['inputs'] = self.inputs_frame.get_data()
        updated_data['optional_inputs'] = self.optionals_frame.get_data()
        updated_data['outputs'] = self.outputs_frame.get_data()
        updated_data['function'] = self.func_name_entry.get()
        updated_data['function_def'] = self.func_def_text.get("1.0", "end-1c").strip()
        return updated_data

class TemplateEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref, theme):
        super().__init__(master, agent_name, agent_data, app_ref, theme)
        tab_view = ctk.CTkTabview(self, fg_color=self.theme['colors']['bg_primary'])
        tab_view.grid(row=0, column=0, sticky="nsew")

        self.create_settings_tab(tab_view.add("Settings"))
        self.create_inputs_tab(tab_view.add("Inputs"))
        self.create_optionals_tab(tab_view.add("Optional Inputs"))
        self.create_outputs_tab(tab_view.add("Outputs"))
        self.create_prompt_tab(tab_view.add("Prompt"))

    def create_settings_tab(self, tab):
        #print("\n" + "="*20 + f" INSTRUMENTATION START: TemplateEditorFrame for '{self.agent_name}' " + "="*20)
        #print(f"STEP 0: Full self.data dictionary received by editor:\n{json.dumps(self.data, indent=2)}")

        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        #print("STEP 1: Tab configured.")

        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'], weight="bold")
        main_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['main_size'])
        #print("STEP 2: Fonts created.")
        
        #print("STEP 3: Creating Agent Name widgets.")
        ctk.CTkLabel(tab, text="Agent Name:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.name_entry = ctk.CTkEntry(tab, font=main_font)
        #print(f"STEP 3a: Inserting agent name '{self.agent_name}' into name_entry.")
        self.name_entry.insert(0, self.agent_name)
        self.name_entry.pack(fill="x", padx=10, pady=5)
        #print("STEP 3b: Agent Name widgets created and populated.")
        
        #print("STEP 4: Creating Help Text widgets.")
        ctk.CTkLabel(tab, text="Help Text:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.help_text = ctk.CTkTextbox(tab, height=200, font=main_font)
        help_data = self.data.get("help", "")
        #print(f"STEP 4a: Inserting help text '{help_data[:50]}...' into help_text.")
        self.help_text.insert("1.0", help_data)
        self.help_text.pack(fill="x", padx=10, pady=5)
        #print("STEP 4b: Help Text widgets created and populated.")

        #print("STEP 5: Creating Web Service Tags widgets.")
        ctk.CTkLabel(tab, text="Web Service Tags (comma-separated):", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.web_services_entry = ctk.CTkEntry(tab, font=main_font)
        tags_list = self.data.get("web_services", [])
        #print(f"STEP 5a: Got web_services list from self.data: {tags_list} (Type: {type(tags_list)})")
        display_string = ", ".join(tags_list)
        #print(f"STEP 5b: Converted list to string: '{display_string}'")
        #print(f"STEP 5c: Inserting string into web_services_entry.")
        self.web_services_entry.insert(0, display_string)
        self.web_services_entry.pack(fill="x", padx=10, pady=5)
        #print("STEP 5d: Web Service Tags widgets created and populated.")
        #print("="*20 + " INSTRUMENTATION END: TemplateEditorFrame " + "="*20 + "\n")


    def create_inputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.inputs_frame = ListEditorFrame(tab, "Required Inputs", self.data.get("inputs", []), self.theme); self.inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_optionals_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.optionals_frame = ListEditorFrame(tab, "Optional Inputs", self.data.get("optional_inputs", []), self.theme); self.optionals_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_outputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.outputs_frame = ListEditorFrame(tab, "Outputs", self.data.get("outputs", []), self.theme); self.outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

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
        
    def get_data(self):
        updated_data = copy.deepcopy(self.data)
        updated_data['name'] = self.name_entry.get().strip()
        updated_data['help'] = self.help_text.get("1.0", "end-1c").strip()
        
        tags_string = self.web_services_entry.get().strip()
        tags_list = [tag.strip() for tag in tags_string.split(',') if tag.strip()]
        if tags_list:
            updated_data['web_services'] = tags_list
        elif 'web_services' in updated_data:
            del updated_data['web_services']
            
        updated_data['inputs'] = self.inputs_frame.get_data()
        updated_data['optional_inputs'] = self.optionals_frame.get_data()
        updated_data['outputs'] = self.outputs_frame.get_data()
        updated_data['prompt'] = self.prompt_text.get("1.0", "end-1c").strip()
        return updated_data

class JsonEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref, theme):
        super().__init__(master, agent_name, agent_data, app_ref, theme)
        
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=16, weight="bold")
        code_font = ctk.CTkFont(family=self.theme['fonts']['code_family'], size=self.theme['fonts']['code_size'])

        ctk.CTkLabel(self, text="Raw JSON Editor", font=label_font).pack(anchor="w", padx=10, pady=(10,0))
        self.textbox = ctk.CTkTextbox(self, font=code_font); self.textbox.pack(fill="both", expand=True, padx=10, pady=10)
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
