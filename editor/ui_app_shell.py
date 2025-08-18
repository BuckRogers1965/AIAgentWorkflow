# --- START OF COMPLETE FILE editor/ui_app_shell.py ---
import customtkinter as ctk
from tkinter import messagebox, Menu
import copy
import json
import re
import tkinter as tk # <-- NEW IMPORT

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

        self.agent_families = {} 

        self.grid_columnconfigure(0, weight=0)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=0)
        
        self.configure(fg_color=self.theme['colors']['bg_primary'])

        self.bind_all("<Control-v>", self.handle_paste_event)
        self.bind_all("<Command-v>", self.handle_paste_event)

        self.create_ui_elements()

        self.focus_set()

    def delete_specific_agent_version(self, agent_full_id):
        if not agent_full_id: return

        dependents = self.config_manager.check_agent_usage(agent_full_id)
        if dependents:
            messagebox.showerror(
                "Cannot Delete Version",
                f"Cannot delete '{agent_full_id}' because it is used by:\n\n- " + "\n- ".join(set(dependents))
            )
            return

        msg = f"Are you sure you want to delete this specific agent version?\n\n{agent_full_id}\n\nThis cannot be undone."
        if not messagebox.askyesno("Confirm Delete", msg, parent=self):
            return

        self.config_manager.delete_agent(agent_full_id)
        self.config_manager.save()

        self.refresh_agent_list()
        self.show_welcome_message()
        self.show_toast(f"Agent version '{agent_full_id}' deleted.")

    def _get_major_version_str(self, full_id: str) -> str | None:
        """
        Extracts the major version string (e.g., 'v1', 'v2') from a full ID.
        Returns None if it's not a versioned name.
        """
        match = re.search(r'/v(\d+)\.\d+$', full_id)
        if match:
            return f"v{match.group(1)}"
        return None

    def _get_versions_grouped_by_major(self, full_ids: list[str]) -> dict:
        """
        Groups a list of full IDs by their major version string.
        Returns a dict like {'v1': ['.../v1.0', '.../v1.1'], 'v2': ['.../v2.0']}
        """
        grouped = {}
        for full_id in full_ids:
            major_str = self._get_major_version_str(full_id)
            if major_str:
                if major_str not in grouped:
                    grouped[major_str] = []
                grouped[major_str].append(full_id)
        return grouped

    def _get_agent_base_name(self, full_id: str) -> str:
        """
        Calculates the base name for an agent, which is the full path without the version.
        - For '/path/name/v1.0', returns '/path/name'.
        - For 'plain_name', returns 'plain_name'.
        """
        match = re.match(r'^(.*)/v(\d+\.\d+)$', full_id)
        if match:
            return match.group(1) # Return the part before the version
        return full_id # It's a plain name, so the base name is the name itself

    def _get_agent_grouping_key(self, full_id: str) -> str:
        match = re.match(r'^(.*)/([^/]+)/v(\d+\.\d+)$', full_id)
        if match:
            return match.group(2)
        return full_id

    def _find_latest_version(self, version_ids: list[str]) -> str:
        latest_version = (-1, -1)
        latest_id = version_ids[0]

        for full_id in version_ids:
            match = re.search(r'/v(\d+)\.(\d+)$', full_id)
            if match:
                major, minor = int(match.group(1)), int(match.group(2))
                if major > latest_version[0] or (major == latest_version[0] and minor >= latest_version[1]):
                    latest_version = (major, minor)
                    latest_id = full_id
        
        return latest_id

    def handle_paste_event(self, event=None):
        if self.current_agent_name is not None:
            return

        try:
            clipboard_content = self.clipboard_get()
            if not clipboard_content.strip():
                return

            import json_repair
            data = json.loads(json_repair.repair_json(clipboard_content))

            agent_definition_to_process = None

            if isinstance(data, dict) and "type" in data:
                agent_definition_to_process = data
            elif isinstance(data, dict) and len(data) == 1:
                first_value = next(iter(data.values()))
                if isinstance(first_value, dict) and "type" in first_value:
                    agent_definition_to_process = first_value

            if agent_definition_to_process:
                self.process_pasted_agent(agent_definition_to_process)
                return "break"
            else:
                self.show_toast("Paste failed: JSON does not appear to be a valid agent definition.", is_error=True)
                return "break"

        except (json.JSONDecodeError, TypeError):
            self.show_toast("Paste failed: Clipboard does not contain valid JSON.", is_error=True)
            return "break"
        except self.tk.TclError:
            self.show_toast("Paste failed: Clipboard does not contain text.", is_error=True)
            return "break"

    def process_pasted_agent(self, agent_data):
        base_name = "pasted_agent"
        try:
            clipboard_text = self.clipboard_get()
            match = re.search(r'"([^"]+)"\s*:\s*\{', clipboard_text.strip())
            if match:
                base_name = match.group(1)
        except Exception:
            pass
        
        if self.config_manager.get_agent_data(base_name):
            new_name = self.config_manager.generate_unique_name(base_name)
        else:
            new_name = base_name

        self.config_manager.update_agent(new_name, agent_data)
        self.config_manager.save()
        
        self.refresh_agent_list()
        self.select_agent(new_name)
        
        self.show_toast(f"New agent created from clipboard: '{new_name}'")

    def _search_entry_paste(self, event):
        try:
            self.search_entry.event_generate("<<Paste>>")
        except:
            pass
        return "break"

    def deselect_search_box(self, event=None):
        self.focus_set()

    def create_ui_elements(self):
        for widget in self.winfo_children():
            widget.destroy()
            
        self.create_agent_list_panel()
        self.create_editor_panel()
        self.create_modal_overlay()
        self.refresh_agent_list()

        self.search_entry.bind("<Control-v>", self._search_entry_paste)
        self.search_entry.bind("<Command-v>", self._search_entry_paste)
        self.agent_list_frame.bind("<Button-1>", self.deselect_search_box)
        
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
        
        clear_search_btn = ctk.CTkButton(search_frame, text="X", width=30, text_color="white", fg_color=self.theme['colors']['error'], command=lambda: self.search_text.set(""))
        clear_search_btn.grid(row=0, column=1, padx=(5, 0))
        
        self.agent_scroll_frame = ctk.CTkScrollableFrame(self.agent_list_frame, fg_color=self.theme['colors']['bg_primary'])
        self.agent_scroll_frame.grid(row=2, column=0, padx=10, pady=(0, 10), sticky="nsew")

    def create_editor_panel(self):
        self.editor_container = ctk.CTkFrame(self, fg_color="transparent")
        self.editor_container.grid(row=0, column=1, padx=10, pady=(10,0), sticky="nsew")
        self.editor_container.grid_rowconfigure(0, weight=1); self.editor_container.grid_columnconfigure(0, weight=1)

        self.editor_container.bind("<Button-1>", self.deselect_search_box)
        
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
        if not self.editor_frame_instance:
            return

        # Get the most up-to-date data from the form fields
        current_agent_data = self.editor_frame_instance.get_data()
        if current_agent_data is None:
            messagebox.showerror("Error", "Cannot run agent. Please check editor for errors.")
            return

        # This part remains the same
        self.show_overlay(f"Running '{self.current_agent_name}'...")
        modal = RunAgentModal(self, self.current_agent_name, current_agent_data, self.core_lib, self.theme)
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
        
        self.agent_families = {}
        for full_id in self.config_manager.get_agent_names():
            grouping_key = self._get_agent_grouping_key(full_id)
            if grouping_key not in self.agent_families:
                self.agent_families[grouping_key] = []
            self.agent_families[grouping_key].append(full_id)

        search_term = self.search_text.get().lower()
        show_hidden = self.show_hidden_agents_var.get() == 1
        
        is_workflow_steps_tab = (isinstance(self.editor_frame_instance, WorkflowEditorFrame) and
                                 self.editor_frame_instance.tab_view.get() == "Steps")
        
        main_font = (self.theme['fonts']['main_family'], self.theme['fonts']['main_size'])

        for display_name, full_ids in sorted(self.agent_families.items()):
            if search_term:
                matches_display_name = search_term in display_name.lower()
                matches_full_id = any(search_term in full_id.lower() for full_id in full_ids)
                if not (matches_display_name or matches_full_id):
                    continue

            latest_id = self._find_latest_version(full_ids)
            agent_data = self.config_manager.get_agent_data(latest_id)

            if agent_data.get("gui", {}).get("hide_in_agent_list", False) and not show_hidden: continue
            
            agent_type = agent_data.get("type", "json")
            prefix = {"workflow": "W", "proc": "P", "template": "T"}.get(agent_type, "J")
            
            row = ctk.CTkFrame(self.agent_scroll_frame, fg_color="transparent")
            row.pack(fill="x", padx=2, pady=2)
            
            btn = ctk.CTkButton(row, text=f"[{prefix}] {display_name}", anchor="w", font=main_font)
            btn.pack(side="left", fill="x", expand=True)

            if is_workflow_steps_tab:
                workflow_editor = self.editor_frame_instance
                # DRAG PAYLOAD IS THE GROUPING KEY (DISPLAY NAME)
                btn.bind("<ButtonPress-1>", lambda e, n=display_name: workflow_editor._on_agent_drag_start(e, n))
            else:
                # REGULAR CLICK ACTION FOR OPENING EDITOR
                btn.bind("<Button-1>", lambda e, n=display_name: self.on_agent_list_click(n, e))

    def on_agent_list_click(self, grouping_key, event):
        versions = self.agent_families.get(grouping_key, [])
        if not versions:
            return

        # If there's only one version AND it's a plain name, open it directly.
        if len(versions) == 1 and not re.search(r'/v\d+\.\d+$', versions[0]):
            self.select_agent(versions[0])
            return
        
        # Create and post the version selection menu
        menu = tk.Menu(self, tearoff=0)

        # Robust sorting key function that handles plain names
        def version_sort_key(full_id):
            match = re.search(r'/v(\d+)\.(\d+)$', full_id)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            else:
                return (-1, -1) # Sort plain names before versioned ones

        sorted_versions = sorted(versions, key=version_sort_key)
        
        for version_id in sorted_versions:
            # Robustly get the display text for the menu item
            version_match = re.search(r'v\d+\.\d+$', version_id)
            if version_match:
                label_text = f"Open {version_match.group(0)}"
            else:
                # This handles the plain name case
                label_text = f"Open '{version_id}'"
            
            menu.add_command(label=label_text, command=lambda v=version_id: self.select_agent(v))
        
        menu.post(event.x_root, event.y_root)

    def add_new_agent(self, choice):
        agent_type = choice.lower()
        new_name = self.config_manager.create_new_agent(agent_type)
        self.config_manager.save()
        self.refresh_agent_list()
        self.select_agent(new_name)
    
    def select_agent(self, full_identifier):
        if full_identifier in self.config_manager.get_agent_names():
            self.current_agent_name = full_identifier
            self.build_editor_form()
        else:
            messagebox.showerror("Error", f"Could not find agent with ID '{full_identifier}'")
            self.show_welcome_message()
        
    def show_welcome_message(self):
        if self.editor_frame_instance: self.editor_frame_instance.destroy()
        self.current_agent_name = None; self.editor_frame_instance = None
        
        welcome_text = (
            "Select an agent to edit or create a new one.\n\n"
            "or\n\n"
            "Paste an agent definition into this window.\n"
            "(Click here to focus)"
        )
        
        label = ctk.CTkLabel(self.editor_container, text=welcome_text, 
                             font=(self.theme['fonts']['main_family'], self.theme['fonts']['title_size']), 
                             text_color=self.theme['colors']['text_secondary'])
        label.place(relx=0.5, rely=0.5, anchor="center")

        label.bind("<Button-1>", self.deselect_search_box)
        
        self.action_bar.grid_remove()
        self.refresh_agent_list()

    def save_agent(self):
        if not self.editor_frame_instance or not self.current_agent_name:
            return

        original_data = self.config_manager.get_agent_data(self.current_agent_name)
        if not original_data:
            messagebox.showerror("Error", "Could not find original agent data to save against.")
            return

        updated_data = self.editor_frame_instance.get_data()
        if updated_data is None: return

        proposed_new_name = updated_data.pop('name', self.current_agent_name)
        
        original_json = json.dumps(original_data, sort_keys=True)
        updated_json = json.dumps(updated_data, sort_keys=True)

        if original_json == updated_json and proposed_new_name == self.current_agent_name:
            self.show_toast("No changes detected. Save cancelled.")
            self.show_welcome_message()
            return

        new_name_is_versioned = bool(re.search(r'/v(\d+)\.(\d+)$', proposed_new_name))
        old_name_is_versioned = bool(re.search(r'/v(\d+)\.(\d+)$', self.current_agent_name))

        # CASE 1: MIGRATING from a plain name to a versioned name OR just renaming a plain name.
        if not old_name_is_versioned:
            if not self.config_manager.rename_agent(self.current_agent_name, proposed_new_name):
                messagebox.showerror("Error", f"Agent name '{proposed_new_name}' already exists.")
                return
            # Now that it's renamed, update its content with the changes from the editor.
            self.config_manager.update_agent(proposed_new_name, updated_data)
            self.config_manager.save()
            toast_message = f"Agent '{self.current_agent_name}' migrated and saved as '{proposed_new_name}'." if new_name_is_versioned else f"Agent '{self.current_agent_name}' renamed to '{proposed_new_name}'."
            self.show_toast(toast_message)
            self.show_welcome_message()
            return

        # CASE 2: UPDATING an existing versioned agent.
        original_interface = {
            "inputs": set(original_data.get("inputs", [])),
            "optional_inputs": set(original_data.get("optional_inputs", [])),
            "outputs": set(original_data.get("outputs", []))
        }
        current_interface = {
            "inputs": set(updated_data.get("inputs", [])),
            "optional_inputs": set(updated_data.get("optional_inputs", [])),
            "outputs": set(updated_data.get("outputs", []))
        }

        is_breaking_change = False
        if current_interface["inputs"] != original_interface["inputs"]: is_breaking_change = True
        if current_interface["outputs"] != original_interface["outputs"]: is_breaking_change = True
        if original_interface["optional_inputs"] - current_interface["optional_inputs"]: is_breaking_change = True
        
        match = re.search(r'/v(\d+)\.(\d+)$', self.current_agent_name)
        base_name_key = self._get_agent_grouping_key(self.current_agent_name)
        self.refresh_agent_list()
        all_versions_for_family = self.agent_families.get(base_name_key, [])
        
        highest_major, highest_minor_for_major = 0, 0
        current_major = int(match.group(1))

        for version_id in all_versions_for_family:
            v_match = re.search(r'/v(\d+)\.(\d+)$', version_id)
            if v_match:
                major, minor = int(v_match.group(1)), int(v_match.group(2))
                if major > highest_major: highest_major = major
                if major == current_major and minor > highest_minor_for_major:
                    highest_minor_for_major = minor

        if is_breaking_change:
            new_major, new_minor = highest_major + 1, 0
            toast_message = f"Breaking change. Saved as new MAJOR v{new_major}.{new_minor}."
        else:
            new_major, new_minor = current_major, highest_minor_for_major + 1
            toast_message = f"Saved as new MINOR v{new_major}.{new_minor}."

        base_name_path = self._get_agent_base_name(self.current_agent_name)
        new_version_id = f"{base_name_path}/v{new_major}.{new_minor}"

        self.config_manager.update_agent(new_version_id, updated_data)
        
        # --- AUTOMATIC PRUNING LOGIC ---
        pruned_count = 0
        policy = self.config_manager.config.get("gui_settings", {}).get("version_retention_policy", 999)
        
        protected_versions = set()
        for agent in self.config_manager.config["agents"].values():
            if agent.get("type") == "workflow":
                for step in agent.get("steps", []):
                    if "agent" in step:
                        protected_versions.add(step["agent"])

        self.refresh_agent_list()
        pruning_base_name_key = self._get_agent_grouping_key(new_version_id)
        all_versions_to_consider = self.agent_families.get(pruning_base_name_key, [])
        
        versions_by_major = {}
        for version_id in all_versions_to_consider:
            v_match = re.search(r'v(\d+)\.(\d+)$', version_id)
            if v_match:
                major = int(v_match.group(1))
                if major not in versions_by_major:
                    versions_by_major[major] = []
                versions_by_major[major].append(version_id)

        for major_v, version_ids in versions_by_major.items():
            unprotected_versions = [v for v in version_ids if v not in protected_versions]
            
            if len(unprotected_versions) > policy:
                def sort_key(full_id):
                    match = re.search(r'v(\d+)\.(\d+)$', full_id)
                    return int(match.group(2))
                
                unprotected_versions.sort(key=sort_key)
                
                versions_to_delete = unprotected_versions[:-policy]
                for v_id in versions_to_delete:
                    self.config_manager.delete_agent(v_id)
                    pruned_count += 1
        
        if pruned_count > 0:
            toast_message += f" Pruned {pruned_count} unused version(s)."
        
        self.config_manager.save()
        self.show_toast(toast_message)
        self.refresh_agent_list()
        self.show_welcome_message()

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

    def show_toast(self, message, is_error=False):
        if is_error:
            fg_color = self.theme['colors']['error']
            text_color = "white"
        else:
            fg_color = self.theme['colors'].get('success', ("#333", "#555"))
            text_color = "white"
            
        toast = ctk.CTkLabel(self, text=message, fg_color=fg_color, text_color=text_color, corner_radius=10, font=("", 14))
        toast.place(relx=0.5, rely=0.95, anchor="center"); toast.lift(); self.after(3500, toast.destroy)
            
    def show_help_modal(self, title, content):
        help_window = ctk.CTkToplevel(self)
        help_window.title(title); help_window.geometry("600x600")
        help_window.transient(self); help_window.grab_set()
        textbox = ctk.CTkTextbox(help_window, wrap="word", font=(self.theme['fonts']['main_family'], self.theme['fonts']['label_size']))
        textbox.pack(fill="both", expand=True, padx=10, pady=10)
        textbox.insert("1.0", content); textbox.configure(state="disabled")
