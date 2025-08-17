# --- START OF FILE workflow_editor.py ---
import customtkinter as ctk
from tkinter import ttk, Menu
import copy
import time
import re
import json

# Import the necessary classes from its module
from ui_editors import BaseEditorFrame, ListEditorFrame, GuiHintsEditorFrame


class VersionSelectorModal(ctk.CTkToplevel):
    """A simple modal to let the user choose a specific agent version to add."""
    def __init__(self, parent, all_versions: list[str]):
        super().__init__(parent)
        self.title("Select Version")
        self.result = None
        
        parent_geo = parent.winfo_geometry().split('+')
        parent_x = int(parent_geo[1])
        parent_y = int(parent_geo[2])
        parent_w = int(parent_geo[0].split('x')[0])
        self.geometry(f"300x{50 + len(all_versions) * 40}+{parent_x + parent_w // 3}+{parent_y + 200}")

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="Choose a version to add to the workflow:").pack(pady=10, padx=10)
        
        # Robust sorting key function that handles plain names
        def version_sort_key(full_id):
            match = re.search(r'/v(\d+)\.(\d+)$', full_id)
            if match:
                return (int(match.group(1)), int(match.group(2)))
            else:
                return (-1, -1)
        
        for version_id in sorted(all_versions, key=version_sort_key):
            version_match = re.search(r'v\d+\.\d+$', version_id)
            if version_match:
                label_text = version_match.group(0)
            else:
                label_text = f"'{version_id}' (no version)"

            btn = ctk.CTkButton(self, text=label_text, command=lambda v=version_id: self.select_version(v))
            btn.pack(pady=5, padx=20, fill="x")

        self.transient(parent)
        self.grab_set()

    def select_version(self, version_id):
        self.result = version_id
        self.destroy()

class MajorVersionSelectorModal(ctk.CTkToplevel):
    """A simple modal to let the user choose a major version branch."""
    def __init__(self, parent, major_versions: list[str]):
        super().__init__(parent)
        self.title("Select Version Branch")
        self.result = None
        
        parent_geo = parent.winfo_geometry().split('+')
        parent_x = int(parent_geo[1])
        parent_y = int(parent_geo[2])
        parent_w = int(parent_geo[0].split('x')[0])
        self.geometry(f"250x{50 + len(major_versions) * 40}+{parent_x + parent_w // 3}+{parent_y + 200}")

        self.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(self, text="This agent has multiple major versions.\nChoose a branch to add:").pack(pady=10, padx=10)
        
        for version_str in sorted(major_versions):
            btn = ctk.CTkButton(self, text=f"Use latest in {version_str}", command=lambda v=version_str: self.select_version(v))
            btn.pack(pady=5, padx=20, fill="x")

        self.transient(parent)
        self.grab_set()

    def select_version(self, version):
        self.result = version
        self.destroy()

class PromoteToWorkflowModal(ctk.CTkToplevel):
    def __init__(self, parent, block_inputs, block_outputs, proposed_name):
        super().__init__(parent)
        self.title("Promote to Workflow")
        self.geometry("600x400")
        
        self.result = None

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        name_frame = ctk.CTkFrame(self)
        name_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        name_frame.grid_columnconfigure(1, weight=1)
        ctk.CTkLabel(name_frame, text="New Workflow Name:").grid(row=0, column=0, padx=5, pady=5)
        self.name_entry = ctk.CTkEntry(name_frame)
        self.name_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")
        self.name_entry.insert(0, proposed_name)

        io_frame = ctk.CTkFrame(self)
        io_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        io_frame.grid_columnconfigure((0, 1), weight=1)
        io_frame.grid_rowconfigure(1, weight=1)
        
        ctk.CTkLabel(io_frame, text="Inputs", font=ctk.CTkFont(weight="bold")).grid(row=0, column=0)
        ctk.CTkLabel(io_frame, text="Outputs", font=ctk.CTkFont(weight="bold")).grid(row=0, column=1)

        inputs_textbox = ctk.CTkTextbox(io_frame)
        inputs_textbox.grid(row=1, column=0, padx=5, pady=5, sticky="nsew")
        inputs_textbox.insert("1.0", "\n".join(block_inputs) or "None")
        inputs_textbox.configure(state="disabled")

        outputs_textbox = ctk.CTkTextbox(io_frame)
        outputs_textbox.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        outputs_textbox.insert("1.0", "\n".join(block_outputs) or "None")
        outputs_textbox.configure(state="disabled")

        button_frame = ctk.CTkFrame(self, fg_color="transparent")
        button_frame.grid(row=2, column=0, padx=10, pady=10, sticky="e")
        ctk.CTkButton(button_frame, text="Cancel", command=self.cancel).pack(side="left", padx=5)
        ctk.CTkButton(button_frame, text="Promote to Workflow", command=self.promote, fg_color="green").pack(side="left", padx=5)
        
        self.transient(parent)
        self.grab_set()

    def promote(self):
        from tkinter import messagebox
        self.result = self.name_entry.get().strip()
        if not self.result:
            messagebox.showerror("Error", "Workflow name cannot be empty.", parent=self)
            return
        self.destroy()

    def cancel(self):
        self.result = None
        self.destroy()

class GuiApi:
    def __init__(self, workflow_editor_frame, source_step_index):
        self.editor = workflow_editor_frame; self.source_index = source_step_index
        self.app_ref = self.editor.app_ref; self.steps = self.editor.data.get("steps", [])
    def add_partner_agent(self, agent_name):
        if not agent_name: return
        agent_info = self.app_ref.config_manager.get_agent_data(agent_name)
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
        return self.app_ref.config_manager.get_agent_data(agent_name) or {}
    def generate_unique_id(self):
        return str(int(time.time()))[-6:]

class WorkflowEditorFrame(BaseEditorFrame):
    def __init__(self, master, agent_name, agent_data, app_ref, theme):
        super().__init__(master, agent_name, agent_data, app_ref, theme)
        
        self._dnd_initialized = False
        self.drag_data = {"source": None, "payload": None, "source_index": -1, "drop_index": -1}
        self.drag_window = None
        self.is_dragging = False
        self._scroll_job_id = None
        self._scroll_direction = None

        self.selected_indices = set()
        self.last_selected_index = None
        self.selected_color = self.theme['colors']['accent_primary']

        self.tab_view = ctk.CTkTabview(self, command=self._on_tab_change, fg_color=self.theme['colors']['bg_primary'])
        self.tab_view.grid(row=0, column=0, sticky="nsew")

        self.bind("<Escape>", self._clear_selection)

        self.create_settings_tab(self.tab_view.add("Settings"))
        self.create_gui_hints_tab(self.tab_view.add("GUI Hints"))
        self.create_inputs_tab(self.tab_view.add("Inputs"))
        self.create_optionals_tab(self.tab_view.add("Optional Inputs"))
        self.create_outputs_tab(self.tab_view.add("Outputs"))
        self.create_steps_tab(self.tab_view.add("Steps"))

    def create_settings_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        label_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['label_size'], weight="bold")
        main_font = ctk.CTkFont(family=self.theme['fonts']['main_family'], size=self.theme['fonts']['main_size'])

        help_btn = self._create_help_button(tab, "Settings for this workflow agent."); help_btn.place(relx=0.98, rely=0.02, anchor="ne")
        ctk.CTkLabel(tab, text="Agent Name:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.name_entry = ctk.CTkEntry(tab, font=main_font); self.name_entry.insert(0, self.agent_name); self.name_entry.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(tab, text="Help Text:", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.help_text = ctk.CTkTextbox(tab, height=200, font=main_font); self.help_text.insert("1.0", self.data.get("help", "")); self.help_text.pack(fill="x", padx=10, pady=5)
        
        ctk.CTkLabel(tab, text="Web Service Tags (comma-separated):", font=label_font).pack(anchor="w", padx=10, pady=(10, 0))
        self.web_services_entry = ctk.CTkEntry(tab, font=main_font)
        self.web_services_entry.insert(0, ", ".join(self.data.get("web_services", [])))
        self.web_services_entry.pack(fill="x", padx=10, pady=5)

        fail_frame = ctk.CTkFrame(tab, fg_color="transparent"); fail_frame.pack(fill="x", padx=10, pady=10)
        self.fail_check_var = ctk.IntVar(value=self.data.get("return_on_fail", 0))
        self.fail_check = ctk.CTkCheckBox(fail_frame, text="Return on Fail", variable=self.fail_check_var, font=main_font); self.fail_check.pack(side="left")

        copy_button = ctk.CTkButton(tab, text="Copy Agent Definition to Clipboard", command=self.copy_agent_definition_to_clipboard)
        copy_button.pack(fill="x", padx=10, pady=(15, 5))

    def create_gui_hints_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        tab.grid_rowconfigure(0, weight=1); tab.grid_columnconfigure(0, weight=1)
        self.gui_hints_frame = GuiHintsEditorFrame(tab, self.data, self.theme, self.app_ref)
        self.gui_hints_frame.grid(row=0, column=0, sticky="nsew")

    def create_inputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.inputs_frame = ListEditorFrame(tab, "Required Inputs", self.data.get("inputs", []), self.theme); self.inputs_frame.pack(fill="both", expand=True, padx=5, pady=5)

    def create_optionals_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.optionals_frame = ListEditorFrame(tab, "Optional Inputs", self.data.get("optional_inputs", []), self.theme); self.optionals_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
    def create_outputs_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        self.outputs_frame = ListEditorFrame(tab, "Outputs", self.data.get("outputs", []), self.theme); self.outputs_frame.pack(fill="both", expand=True, padx=5, pady=5)
        
    def _on_tab_change(self):
        self.app_ref.refresh_agent_list()
        if self.tab_view.get() == "Steps" and not self._dnd_initialized: self._initialize_dnd_features()

    def _initialize_dnd_features(self):
        self.drop_indicator = ttk.Separator(self.steps_frame, orient='horizontal'); self._dnd_initialized = True

    def _create_drag_window(self, text, event):
        if self.drag_window: self.drag_window.destroy()
        self.drag_window = ctk.CTkToplevel(self); self.drag_window.overrideredirect(True)
        self.drag_window.attributes("-alpha", 0.75)
        label = ctk.CTkLabel(self.drag_window, text=text, fg_color="gray20", corner_radius=6, padx=10, pady=5); label.pack()
        self.drag_window.geometry(f"+{event.x_root + 10}+{event.y_root + 10}")

    def _start_drag_polling(self):
        self.is_dragging = True; self.winfo_toplevel().bind("<ButtonRelease-1>", self._on_drop)
        self.winfo_toplevel().bind("<Escape>", self._on_drag_cancel); self._drag_polling_loop()

    def _drag_polling_loop(self):
        if not self.is_dragging: return
        x_root, y_root = self.winfo_pointerx(), self.winfo_pointery()
        if self.drag_window: self.drag_window.geometry(f"+{x_root + 10}+{y_root + 10}")
        canvas = self.steps_frame._parent_canvas
        target_x, target_y, target_w, target_h = canvas.winfo_rootx(), canvas.winfo_rooty(), canvas.winfo_width(), canvas.winfo_height()
        if target_x < x_root < target_x + target_w and target_y < y_root < target_y + target_h:
            y_local = y_root - target_y; self._update_drop_indicator(y_root); self._manage_autoscroll(y_local, target_h)
        else:
            self.drag_data["drop_index"] = -1; self.drop_indicator.place_forget(); self._stop_scroll()
        self.after(20, self._drag_polling_loop)

    def _on_drop(self, event):
        if not self.is_dragging: return
        self.is_dragging = False
        self.winfo_toplevel().unbind("<ButtonRelease-1>")
        self.winfo_toplevel().unbind("<Escape>")
        if self.drag_window: self.drag_window.destroy(); self.drag_window = None
        self._stop_scroll()
        self.drop_indicator.place_forget()

        app = self.winfo_toplevel()

        if self.drag_data["source"] == "agent_list" and self.drag_data["drop_index"] != -1:
            grouping_key = self.drag_data["payload"]
            drop_index = self.drag_data["drop_index"]
            
            full_ids = app.agent_families.get(grouping_key, [])

            if not full_ids:
                app.show_toast(f"Could not find agent '{grouping_key}'", is_error=True)
            
            elif len(full_ids) == 1:
                # If there's only one option, just insert it directly.
                self.insert_agent_as_step(full_ids[0], drop_index)

            else:
                # If there are multiple options, show the selector modal.
                modal = VersionSelectorModal(self, full_ids)
                self.wait_window(modal)
                chosen_version_id = modal.result
                
                if chosen_version_id:
                    self.insert_agent_as_step(chosen_version_id, drop_index)
        
        elif self.drag_data["source"] == "steps_frame" and self.drag_data["drop_index"] != -1:
            self.move_step_to_index(self.drag_data["source_index"], self.drag_data["drop_index"])
            
        self.refresh_steps_list()
        self.drag_data = {"source": None, "payload": None, "source_index": -1, "drop_index": -1}
        self.winfo_toplevel().configure(cursor="")

    def _on_drag_cancel(self, event=None):
        if not self.is_dragging: return
        self.is_dragging = False; self.winfo_toplevel().unbind("<ButtonRelease-1>"); self.winfo_toplevel().unbind("<Escape>")
        if self.drag_window: self.drag_window.destroy(); self.drag_window = None
        self._stop_scroll(); self.drop_indicator.place_forget()
        self.refresh_steps_list(); self.drag_data = {"source": None, "payload": None, "source_index": -1, "drop_index": -1}
        self.winfo_toplevel().configure(cursor="")

    def _on_agent_drag_start(self, event, agent_name):
        if not self._dnd_initialized or self.selected_indices: return
        self.drag_data = {"source": "agent_list", "payload": agent_name, "source_index": -1, "drop_index": -1}
        self.winfo_toplevel().configure(cursor="hand2"); self._create_drag_window(f"[Agent] {agent_name}", event); self._start_drag_polling()

    def _on_step_drag_start(self, event, index, widget):
        if not self._dnd_initialized or self.selected_indices: return
        agent_name = self.data['steps'][index].get('agent', 'Unknown')
        self.drag_data = {"source": "steps_frame", "payload": self.data['steps'][index], "source_index": index, "drop_index": -1}
        widget.pack_forget(); self.winfo_toplevel().configure(cursor="fleur"); self._create_drag_window(f"Step {index}: {agent_name}", event); self._start_drag_polling()

    def _update_drop_indicator(self, root_y):
        try:
            canvas = self.steps_frame._parent_canvas; y_on_canvas = root_y - canvas.winfo_rooty(); y_content = canvas.canvasy(y_on_canvas)
            children = [child for child in self.steps_frame.winfo_children() if child is not self.drop_indicator]
            new_index = len(children)
            for i, child in enumerate(children):
                if y_content < child.winfo_y() + child.winfo_height() / 2: new_index = i; break
            self.drag_data["drop_index"] = new_index
            if self.drag_data["source"] == "steps_frame" and (new_index == self.drag_data["source_index"] or new_index == self.drag_data["source_index"] + 1):
                 self.drop_indicator.place_forget(); self.drag_data["drop_index"] = -1; return
            if new_index == 0: indicator_y = 0
            elif new_index >= len(children):
                last_child = children[-1] if children else None; indicator_y = last_child.winfo_y() + last_child.winfo_height() if last_child else 0
            else:
                prev_child = children[new_index - 1]; indicator_y = prev_child.winfo_y() + prev_child.winfo_height() + 1
            self.drop_indicator.place(x=0, y=indicator_y, relwidth=1, height=2)
        except Exception: self.drop_indicator.place_forget()

    def _manage_autoscroll(self, y_local, widget_height):
        scroll_threshold = 40
        if y_local < scroll_threshold: self._start_scroll("up")
        elif y_local > widget_height - scroll_threshold: self._start_scroll("down")
        else: self._stop_scroll()

    def _start_scroll(self, direction):
        if self._scroll_direction == direction: return
        self._stop_scroll(); self._scroll_direction = direction; delta = -1 if direction == "up" else 1
        def scroll_action():
            if not self.is_dragging: self._stop_scroll(); return
            self.steps_frame._parent_canvas.yview_scroll(delta, "units"); self.update_idletasks()
            self._update_drop_indicator(self.winfo_pointery()); self._scroll_job_id = self.after(30, scroll_action)
        scroll_action()

    def _stop_scroll(self):
        if self._scroll_job_id: self.after_cancel(self._scroll_job_id); self._scroll_job_id = None
        self._scroll_direction = None
    
    def _clear_selection(self, event=None):
        self.selected_indices.clear()
        self.last_selected_index = None
        self.refresh_steps_list()
        return "break"

    def _on_step_select_click(self, event, index):
        self.focus_set()
        if event.state & 4: # Control key mask
            self.selected_indices = {index}; self.last_selected_index = index
        elif event.state & 1: # Shift key mask
            if self.last_selected_index is not None:
                start, end = min(self.last_selected_index, index), max(self.last_selected_index, index)
                self.selected_indices = set(range(start, end + 1))
            else: 
                self.selected_indices = {index}; self.last_selected_index = index
        else: # Normal click (no modifier)
             self.selected_indices = {index}; self.last_selected_index = index
        self.refresh_steps_list(); return "break"

    def _on_step_right_click(self, event, index):
        if index in self.selected_indices:
            self._open_promote_dialog()

    def _open_promote_dialog(self):
        all_steps = self.data.get("steps", [])
        min_idx, max_idx = min(self.selected_indices), max(self.selected_indices)

        vars_before = set(self.data.get("inputs", []))
        for i in range(min_idx): vars_before.update(all_steps[i].get("output", []))
        vars_after = set()
        for i in range(max_idx + 1, len(all_steps)):
            for val in all_steps[i].get("params", {}).values(): vars_after.update(re.findall(r'\$(\w+)', str(val)))
        vars_inside = set()
        for i in range(min_idx, max_idx + 1): vars_inside.update(all_steps[i].get("output", []))
        vars_used_inside = set()
        for i in range(min_idx, max_idx + 1):
            for val in all_steps[i].get("params", {}).values(): vars_used_inside.update(re.findall(r'\$(\w+)', str(val)))
        
        block_inputs = sorted(list(vars_used_inside.intersection(vars_before)))
        block_outputs = sorted(list(vars_inside.intersection(vars_after)))
        
        proposed_name = f"promoted_workflow_{int(time.time()) % 10000}"
        
        modal = PromoteToWorkflowModal(self, block_inputs, block_outputs, proposed_name)
        self.wait_window(modal)
        
        new_name = modal.result
        if new_name:
            self._perform_refactoring(new_name, block_inputs, block_outputs)
        
        self._clear_selection()

    def _perform_refactoring(self, new_name, block_inputs, block_outputs):
        from tkinter import messagebox
        if self.app_ref.config_manager.get_agent_data(new_name):
            messagebox.showerror("Error", f"An agent named '{new_name}' already exists.")
            return

        min_idx, max_idx = min(self.selected_indices), max(self.selected_indices)
        block_steps = copy.deepcopy(self.data["steps"][min_idx : max_idx + 1])
        
        new_workflow_data = {
            "type": "workflow", "help": f"Promoted from a block of steps in '{self.agent_name}'.",
            "inputs": block_inputs, "optional_inputs": [], "outputs": block_outputs,
            "steps": block_steps, "return_on_fail": True
        }

        self.app_ref.config_manager.config['agents'][new_name] = new_workflow_data

        replacement_step = {
            "agent": new_name,
            "params": {key: f"${key}" for key in block_inputs},
            "output": block_outputs
        }

        original_steps = self.data.get("steps", [])
        new_steps = original_steps[:min_idx] + [replacement_step] + original_steps[max_idx+1:]
        self.data["steps"] = new_steps

        self.app_ref.config_manager.save()
        self.app_ref.refresh_agent_list()
        self.refresh_steps_list()
        self.app_ref.show_toast(f"Block promoted to new workflow: '{new_name}'")

    def create_steps_tab(self, tab):
        tab.configure(fg_color=self.theme['colors']['bg_secondary'])
        tab.grid_rowconfigure(1, weight=1); tab.grid_columnconfigure(0, weight=1)
        
        help_btn = self._create_help_button(tab, "This is the core of the workflow.")
        help_btn.place(relx=0.98, rely=0.02, anchor="ne")
        
        ctk.CTkLabel(tab, text="Workflow Steps").grid(row=0, column=0, pady=(5,0))
        self.steps_frame = ctk.CTkScrollableFrame(tab, fg_color=self.theme['colors']['bg_primary'])
        self.steps_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        self.steps_frame.bind("<Button-1>", self._clear_selection)
        self.steps_frame._parent_canvas.bind("<Button-1>", self._clear_selection)
        self.refresh_steps_list()

    def move_step_to_index(self, source_index, dest_index):
        steps = self.data.get('steps', [])
        if not (0 <= source_index < len(steps)): return
        item_to_move = steps.pop(source_index)
        if dest_index > source_index: dest_index -= 1
        dest_index = max(0, min(dest_index, len(steps))); steps.insert(dest_index, item_to_move)

    def remove_step(self, index): self.data['steps'].pop(index); self.refresh_steps_list()
    
    def add_agent_as_step(self, agent_name): self.insert_agent_as_step(agent_name, len(self.data.get("steps", [])))

    def insert_agent_as_step(self, agent_name, index):
        agent_info = self.app_ref.config_manager.get_agent_data(agent_name)
        if not agent_info: return
        new_params = {key: f"${key}" for key in agent_info.get("inputs", [])}
        new_step = {"agent": agent_name, "params": new_params, "output": agent_info.get("outputs", ["output"]).copy()}
        self.data.setdefault("steps", []).insert(index, new_step)
        if agent_info.get("gui", {}).get("on_add"): self._process_gui_directives(agent_info["gui"], index)

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

    def refresh_steps_list(self):
        if hasattr(self, 'drop_indicator') and self.drop_indicator.winfo_ismapped():
             self.drop_indicator.place_forget()
        for widget in self.steps_frame.winfo_children():
            if not hasattr(self, 'drop_indicator') or widget is not self.drop_indicator:
                widget.destroy()

        steps = self.data.get("steps", []); current_indent = 0; indent_char = "    "
        main_font = (self.theme['fonts']['main_family'], self.theme['fonts']['main_size'])

        for i, step in enumerate(steps):
            agent_name = step.get('agent', 'Unknown'); agent_def = self.app_ref.config_manager.get_agent_data(agent_name)
            gui_hints = (agent_def or {}).get("gui", {}); current_indent = max(0, current_indent + gui_hints.get("indent_before", 0))
            is_selected = i in self.selected_indices
            frame_color = self.selected_color if is_selected else "transparent"
            step_frame = ctk.CTkFrame(self.steps_frame, fg_color=frame_color); step_frame.pack(fill="x", pady=2)
            step_frame.grid_columnconfigure(1, weight=1)
            
            edit_btn = ctk.CTkButton(step_frame, text="Edit", width=60, command=lambda index=i: self.app_ref.open_step_editor(index)); edit_btn.grid(row=0, column=0, padx=5, pady=5)
            label_text = f"{indent_char * current_indent}{i}. {agent_name}"
            step_label = ctk.CTkLabel(step_frame, text=label_text, font=main_font)
            step_label.grid(row=0, column=1, padx=10, pady=5, sticky="w")
            if not agent_def: step_label.configure(text_color="red", text=f"{label_text} (not found)")
            remove_btn = ctk.CTkButton(step_frame, text="X", width=30, fg_color=self.theme['colors']['error'], command=lambda index=i: self.remove_step(index)); remove_btn.grid(row=0, column=2, padx=5, pady=5)
            
            for widget in [step_frame, step_label]:
                widget.bind("<ButtonPress-1>", lambda e, index=i, w=step_frame: self._on_step_drag_start(e, index, w))
                widget.bind("<Control-Button-1>", lambda e, index=i: self._on_step_select_click(e, index))
                widget.bind("<Command-Button-1>", lambda e, index=i: self._on_step_select_click(e, index))
                widget.bind("<Shift-Button-1>", lambda e, index=i: self._on_step_select_click(e, index))
                widget.bind("<Button-2>", lambda e, index=i: self._on_step_right_click(e, index))
                widget.bind("<Button-3>", lambda e, index=i: self._on_step_right_click(e, index))
            current_indent = max(0, current_indent + gui_hints.get("indent_after", 0))
            
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
        updated_data['return_on_fail'] = self.fail_check_var.get()

        # Handle GUI data
        param_hints = self.gui_hints_frame.get_data()
        if param_hints:
            updated_data.setdefault('gui', {})['param_hints'] = param_hints
        elif 'gui' in updated_data and 'param_hints' in updated_data['gui']:
            del updated_data['gui']['param_hints']
            if not updated_data['gui']:
                del updated_data['gui']
                
        return updated_data
