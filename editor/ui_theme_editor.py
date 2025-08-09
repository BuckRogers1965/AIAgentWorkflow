# --- START OF FILE ui_theme_editor.py ---
import customtkinter as ctk
from tkinter import messagebox, colorchooser
import json
import copy
import os

class ThemeEditorModal(ctk.CTkToplevel):
    def __init__(self, parent, theme_manager):
        super().__init__(parent)
        self.title("Theme Editor")
        self.geometry("800x600")
        self.app_ref = parent
        self.theme_manager = theme_manager
        self.current_theme_name = self.theme_manager.current_theme_name
        self.theme_data = copy.deepcopy(self.theme_manager.get_current_theme_data())
        self.entries = {}
        
        # --- FONT MANAGER IS GONE. USE A SIMPLE, RELIABLE LIST. ---
        self.system_fonts = [
            "Arial", "Courier New", "Georgia", "Helvetica", 
            "SF Mono", "SF Pro Text", "Times New Roman", "Verdana"
        ]
        # --- END OF FIX ---

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(1, weight=1)

        self.create_widgets()
        self.populate_fields()
        
        self.transient(parent)
        self.grab_set()

    # ... (The rest of the file is IDENTICAL to the last working version,
    #      but now it uses the safe, hard-coded self.system_fonts list)
    def create_widgets(self):
        header_frame = ctk.CTkFrame(self)
        header_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
        ctk.CTkLabel(header_frame, text="Current Theme:").pack(side="left", padx=5)
        self.theme_menu = ctk.CTkOptionMenu(header_frame, values=self.theme_manager.get_theme_names(), command=self.on_theme_select)
        self.theme_menu.set(self.current_theme_name)
        self.theme_menu.pack(side="left", padx=5)
        ctk.CTkButton(header_frame, text="Delete Theme", fg_color="red", command=self.delete_theme).pack(side="right", padx=5)

        self.scroll_frame = ctk.CTkScrollableFrame(self)
        self.scroll_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")
        self.scroll_frame.grid_columnconfigure(1, weight=1)

        footer_frame = ctk.CTkFrame(self)
        footer_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")
        ctk.CTkButton(footer_frame, text="Cancel", command=self.destroy).pack(side="right", padx=5)
        ctk.CTkButton(footer_frame, text="Save As New Theme...", command=self.save_as_new).pack(side="right", padx=5)
        ctk.CTkButton(footer_frame, text="Save and Apply", fg_color="green", command=self.save_and_apply).pack(side="right", padx=5)

    def populate_fields(self):
        for widget in self.scroll_frame.winfo_children():
            widget.destroy()
        self.entries.clear()
        
        row = 0
        for category, items in self.theme_data.items():
            ctk.CTkLabel(self.scroll_frame, text=category.replace("_", " ").title(), font=ctk.CTkFont(weight="bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(10, 5), padx=5)
            row += 1
            for key, value in items.items():
                label_text = key.replace("_", " ").title()
                ctk.CTkLabel(self.scroll_frame, text=label_text).grid(row=row, column=0, sticky="w", padx=10)
                
                if category == "fonts" and "family" in key:
                    font_var = ctk.StringVar(value=str(value))
                    display_fonts = sorted(self.system_fonts)
                    if str(value) not in display_fonts:
                        display_fonts.insert(0, str(value))
                    font_menu = ctk.CTkOptionMenu(self.scroll_frame, variable=font_var, values=display_fonts)
                    font_menu.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
                    self.entries[f"{category}.{key}"] = font_var
                else:
                    entry = ctk.CTkEntry(self.scroll_frame)
                    entry.insert(0, str(value))
                    entry.grid(row=row, column=1, sticky="ew", padx=5, pady=2)
                    self.entries[f"{category}.{key}"] = entry
                
                if category == "colors":
                    color_btn = ctk.CTkButton(self.scroll_frame, text="", width=28, height=28, border_width=1, command=lambda e=self.entries[f"{category}.{key}"]: self.pick_color(e))
                    try:
                        color_btn.configure(fg_color=str(value), border_color="white")
                    except Exception:
                        color_btn.configure(fg_color="black", border_color="red")
                    color_btn.grid(row=row, column=2, padx=5)
                row += 1

    def pick_color(self, entry_widget):
        try:
            initial_color = entry_widget.get()
        except:
            initial_color = "#ffffff"
        color_code = colorchooser.askcolor(title="Choose color", initialcolor=initial_color)
        if color_code and color_code[1]:
            if isinstance(entry_widget, ctk.CTkEntry):
                entry_widget.delete(0, "end")
                entry_widget.insert(0, color_code[1])
            self.preview_theme()

    def on_theme_select(self, theme_name):
        self.current_theme_name = theme_name
        self.theme_manager.set_current_theme(theme_name)
        self.theme_data = copy.deepcopy(self.theme_manager.get_current_theme_data())
        self.populate_fields()
        self.app_ref.apply_theme()

    def _get_data_from_form(self):
        new_data = copy.deepcopy(self.theme_data)
        for key_path, widget in self.entries.items():
            category, key = key_path.split('.')
            value = widget.get()
            if category in ["fonts", "styles"] and "size" in key and value.isdigit():
                new_data[category][key] = int(value)
            else:
                new_data[category][key] = value
        return new_data

    def save_and_apply(self):
        new_theme_data = self._get_data_from_form()
        self.theme_manager.save_theme(self.current_theme_name, new_theme_data)
        self.app_ref.apply_theme()
        self.destroy()

    def save_as_new(self):
        dialog = ctk.CTkInputDialog(text="Enter new theme name:", title="Save As")
        new_name = dialog.get_input()
        if new_name and new_name.strip():
            new_name = new_name.strip()
            if new_name in self.theme_manager.get_theme_names():
                messagebox.showerror("Error", "A theme with this name already exists.", parent=self)
                return
            new_theme_data = self._get_data_from_form()
            self.current_theme_name = new_name
            self.theme_manager.save_theme(new_name, new_theme_data)
            self.theme_manager.set_current_theme(new_name)
            self.app_ref.apply_theme()
            self.destroy()
            
    def delete_theme(self):
        if "Default" in self.current_theme_name:
            messagebox.showerror("Error", "Cannot delete a default theme.", parent=self)
            return
        if messagebox.askyesno("Confirm", f"Are you sure you want to delete the theme '{self.current_theme_name}'?", parent=self):
            self.theme_manager.delete_theme(self.current_theme_name)
            self.app_ref.apply_theme()
            self.destroy()

    def preview_theme(self):
        preview_data = self._get_data_from_form()
        self.theme_manager.themes[self.current_theme_name] = preview_data
        self.app_ref.apply_theme()