# --- START OF FILE ui_app_shell.py ---
import customtkinter as ctk
from tkinter import messagebox
import copy
import json

from theme_manager import ThemeManager
from ui_theme_editor import ThemeEditorModal
from config_manager import ConfigManager
from workflow_editor import WorkflowEditorFrame
from ui_editors import ProcEditorFrame, TemplateEditorFrame, JsonEditorFrame
from ui_run_modal import RunAgentModal
from ui_step_editor import StepEditorModal

class GlobalConfigEditorModal(ctk.CTkToplevel):
    def __init__(self, parent, config_manager, theme):
        super().__init__(parent)
        self.title("Global Configuration Editor")
        self.geometry("800x600")
        self.config_manager = config_manager
        self.theme = theme
        self.saved = False
        self.config_data = copy.deepcopy(config_manager.config)
        if "agents" in self.config_data: del self.config_data["agents"]
        self.create_widgets()
        self.transient(parent)
        self.grab_set()
        
    def create_widgets(self):
        self.configure(fg_color=self.theme['colors']['bg_primary'])
        content_frame = ctk.CTkFrame(self, fg_color=self.theme['colors']['bg_secondary'])
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        main_font = (self.theme['fonts']['main_family'], self.theme['fonts']['main_size'])
        code_font = (self.theme['fonts']['editor_code_family'], self.theme['fonts']['editor_code_size'])


        ctk.CTkLabel(content_frame, text="Global Configuration (JSON)", font=(self.theme['fonts']['main_family'], 16, "bold")).pack(pady=(0, 10))
        ctk.CTkLabel(content_frame, text="⚠️ Warning: This editor modifies all configuration except agents. Edit carefully!", text_color=self.theme['colors']['warning']).pack(pady=(0, 10))
        
        self.json_textbox = ctk.CTkTextbox(content_frame, font=code_font)
        self.json_textbox.pack(fill="both", expand=True, padx=5, pady=5)
        self.json_textbox.insert("1.0", json.dumps(self.config_data, indent=2))
        
        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.pack(fill="x", padx=10, pady=10)
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=10)
        ctk.CTkButton(button_frame, text="Save Configuration", command=self.save, fg_color=self.theme['colors']['success']).pack(side="right", padx=10)

    def save(self):
        try:
            new_config = json.loads(self.json_textbox.get("1.0", "end-1c"))
            if "agents" in new_config: messagebox.showerror("Error", "Cannot modify 'agents' key through this editor!"); return
            agents_backup = self.config_manager.config.get("agents", {}); self.config_manager.config = new_config
            self.config_manager.config["agents"] = agents_backup; self.saved = True; self.destroy()
        except json.JSONDecodeError as e: messagebox.showerror("JSON Error", f"Invalid JSON: {e}")
        except Exception as e: messagebox.showerror("Error", f"Failed to save configuration: {e}")
    def cancel(self): self.saved = False; self.destroy()

class App(ctk.CTk):
    def __init__(self, config_path="config.json", engine_loaded=True, core_lib=None):
        super().__init__()
        
        self.theme_manager = ThemeManager()
        self.theme = self.theme_manager.get_current_theme_data()
        
        self.title("Agent Workflow Editor")
        self.geometry("1400x800")
        
        self.config_manager = ConfigManager(config_path=config_path)
        self.core_lib = core_lib
        self.engine_loaded = self.core_lib is not None
        
        self.current_agent_name = None
        self.editor_frame_instance = None
        self.search_text = ctk.StringVar()
        self.search_text.trace("w", self.on_search_changed)
        self.show_hidden_agents_var = ctk.IntVar(value=0)

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        
        self.configure(fg_color=self.theme['colors']['bg_primary'])

        self.create_ui_elements()

    def create_ui_elements(self):
        for widget in self.winfo_children():
            widget.destroy()
            
        self.create_agent_list_panel()
        self.create_editor_panel()
        self.create_modal_overlay()
        self.refresh_agent_list()
        
        if self.current_agent_name:
            self.build_editor_form()
        else:
            self.show_welcome_message()

    def apply_theme(self):
        self.theme = self.theme_manager.get_current_theme_data()
        self.create_ui_elements()

    def create_agent_list_panel(self):
        self.agent_list_frame = ctk.CTkFrame(self, width=320, fg_color=self.theme['colors']['bg_secondary'])
        self.agent_list_frame.grid(row=0, column=0, rowspan=2, padx=10, pady=10, sticky="ns")
        self.agent_list_frame.grid_propagate(False)
        self.agent_list_frame.grid_rowconfigure(2, weight=1)
        self.agent_list_frame.grid_columnconfigure(0, weight=1)
        
        header_frame = ctk.CTkFrame(self.agent_list_frame, fg_color="transparent")
        header_frame.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="ew")
        header_frame.grid_columnconfigure(1, weight=1)
        
        ctk.CTkLabel(header_frame, text="Agents", font=(self.theme['fonts']['main_family'], self.theme['fonts']['title_size'], "bold"), text_color=self.theme['colors']['text_primary']).grid(row=0, column=0, sticky="w")
        
        hidden_checkbox = ctk.CTkCheckBox(
            header_frame, text="", variable=self.show_hidden_agents_var, command=self.refresh_agent_list, 
            width=14, height=14, border_color=self.theme['colors']['bg_secondary'], hover=False
        )
        hidden_checkbox.grid(row=0, column=1, padx=(5,0), sticky="w")
        
        control_frame = ctk.CTkFrame(header_frame, fg_color="transparent")
        control_frame.grid(row=0, column=2, sticky="e")
        
        add_menu = ctk.CTkOptionMenu(control_frame, width=80, values=["Workflow", "Proc", "Template"], command=self.add_new_agent)
        add_menu.pack(side="left", padx=(0,5)); add_menu.set("New")

        config_btn = ctk.CTkButton(control_frame, text="⚙️", width=30, command=self.open_global_config)
        config_btn.pack(side="left", padx=(0,5))
        
        theme_btn = ctk.CTkButton(control_frame, text="T", width=30, command=self.open_theme_editor)
        theme_btn.pack(side="left", padx=(0,0))
        
        search_frame = ctk.CTkFrame(self.agent_list_frame, fg_color="transparent")
        search_frame.grid(row=1, column=0, padx=10, pady=(5, 10), sticky="ew")
        search_frame.grid_columnconfigure(0, weight=1)
        
        self.search_entry = ctk.CTkEntry(search_frame, textvariable=self.search_text, placeholder_text="Filter agents...")
        self.search_entry.grid(row=0, column=0, sticky="ew")
        
        # --- THIS IS THE RESTORED "X" BUTTON ---
        clear_search_btn = ctk.CTkButton(search_frame, text="X", width=30, text_color="white", fg_color=self.theme['colors']['error'], command=lambda: self.search_text.set(""))
        clear_search_btn.grid(row=0, column=1, padx=(5, 0))
        # --- END OF RESTORED BUTTON ---
        
        self.agent_scroll_frame = ctk.CTkScrollableFrame(self.agent_list_frame, fg_color=self.theme['colors']['bg_primary'])
        self.agent_scroll_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

    def create_editor_panel(self):
        self.editor_container = ctk.CTkFrame(self, fg_color="transparent")
        self.editor_container.grid(row=0, column=1, padx=10, pady=(10,0), sticky="nsew")
        self.editor_container.grid_rowconfigure(0, weight=1); self.editor_container.grid_columnconfigure(0, weight=1)
        
        self.action_bar = ctk.CTkFrame(self, fg_color="transparent")
        self.action_bar.grid(row=1, column=1, padx=10, pady=(5, 10), sticky="ew")
        
        self.save_btn = ctk.CTkButton(self.action_bar, text="Save Agent", command=self.save_agent, fg_color=self.theme['colors']['success'])
        self.run_btn = ctk.CTkButton(self.action_bar, text="▶️ Run", command=self.open_run_modal, fg_color=self.theme['colors']['warning'], text_color="#000000")
        
        if not self.engine_loaded:
            self.run_btn.configure(state="disabled")
            
        self.cancel_btn = ctk.CTkButton(self.action_bar, text="Cancel", command=self.show_welcome_message, fg_color=self.theme['colors']['error'])
        self.spacer_frame = ctk.CTkFrame(self.action_bar, fg_color="transparent", width=240, height=40)

    def open_theme_editor(self):
        modal = ThemeEditorModal(self, self.theme_manager)

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
        
        self.editor_frame_instance = editor_class(
            self.editor_container, self.current_agent_name, copy.deepcopy(agent_data), self, theme=self.theme
        )
        self.editor_frame_instance.grid(row=0, column=0, sticky="nsew")
        
        self.refresh_agent_list()
        
    def open_run_modal(self):
        if not self.editor_frame_instance: return
        current_agent_data = self.editor_frame_instance.get_data()
        if current_agent_data is None:
            messagebox.showerror("Error", "Cannot run agent. Please check editor for errors.")
            return

        self.show_overlay(f"Running '{self.current_agent_name}'...")
        modal = RunAgentModal(self, self.current_agent_name, current_agent_data, self.core_lib, self.theme)
        #modal = RunAgentModal(self, self.current_agent_name, current_agent_data, self.core_lib)
        self.wait_window(modal)
        self.hide_overlay()

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
        if hasattr(self, 'agent_scroll_frame'):
            self.agent_scroll_frame._parent_canvas.yview_moveto(0)
            self.agent_scroll_frame.update_idletasks()
        
        for widget in self.agent_scroll_frame.winfo_children(): widget.destroy()
        search_term = self.search_text.get().lower()
        show_hidden = self.show_hidden_agents_var.get() == 1
        
        is_workflow_steps_tab = (isinstance(self.editor_frame_instance, WorkflowEditorFrame) and
                                 self.editor_frame_instance.tab_view.get() == "Steps")
        
        main_font = (self.theme['fonts']['main_family'], self.theme['fonts']['main_size'])

        for name in self.config_manager.get_agent_names():
            agent_data = self.config_manager.get_agent_data(name)
            if agent_data.get("gui", {}).get("hide_in_agent_list", False) and not show_hidden: continue
            if search_term and search_term not in name.lower(): continue
            
            agent_type = agent_data.get("type", "json")
            prefix = {"workflow": "W", "proc": "P", "template": "T"}.get(agent_type, "J")
            
            row = ctk.CTkFrame(self.agent_scroll_frame, fg_color="transparent")
            row.pack(fill="x", padx=2, pady=2)
            
            del_btn = ctk.CTkButton(row, text="X", width=30, fg_color=self.theme['colors']['error'], command=lambda n=name: self.delete_agent(agent_name=n, confirm=True))
            del_btn.pack(side="right")
            
            btn = ctk.CTkButton(row, text=f"[{prefix}] {name}", anchor="w", font=main_font)
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
        label = ctk.CTkLabel(self.editor_container, text="Select an agent to edit or create a new one.", 
                             font=(self.theme['fonts']['main_family'], self.theme['fonts']['title_size']), 
                             text_color=self.theme['colors']['text_secondary'])
        label.place(relx=0.5, rely=0.5, anchor="center")
        self.action_bar.grid_remove()
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
        modal = GlobalConfigEditorModal(self, self.config_manager, self.theme); self.wait_window(modal)
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
        modal = StepEditorModal(self, index, step_data, agent_def, parent_workflow_data, self.theme)
        self.wait_window(modal)
        self.hide_overlay()
        
        if modal.saved:
            self.editor_frame_instance.data["steps"][index] = modal.get_result()
            self.editor_frame_instance.refresh_steps_list()
            
    def show_toast(self, message):
        toast = ctk.CTkLabel(self, text=message, fg_color=("#333", "#555"), text_color="white", corner_radius=10, font=("", 14))
        toast.place(relx=0.5, rely=0.95, anchor="center"); toast.lift(); self.after(2500, toast.destroy)
        
    def show_help_modal(self, title, content):
        help_window = ctk.CTkToplevel(self)
        help_window.title(title); help_window.geometry("600x600")
        help_window.transient(self); help_window.grab_set()
        textbox = ctk.CTkTextbox(help_window, wrap="word", font=(self.theme['fonts']['main_family'], self.theme['fonts']['label_size']))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)
        textbox.insert("1.0", content); textbox.configure(state="disabled")
