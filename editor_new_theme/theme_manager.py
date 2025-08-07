# --- START OF FILE theme_manager.py ---
import json
import os
import customtkinter as ctk

class ThemeManager:
    def __init__(self, theme_file="themes.json"):
        self.file_path = theme_file
        self.data = self._load_themes()
        self.current_theme_name = self.data.get("current_theme", "Default Dark")
        self.themes = self.data.get("themes", {})
        
        # Ensure default themes exist if the file is new or empty
        if "Default Dark" not in self.themes:
            self.themes["Default Dark"] = self._get_default_dark_theme()
        if "Default Light" not in self.themes:
            self.themes["Default Light"] = self._get_default_light_theme()

        self.set_current_theme(self.current_theme_name)

    def _load_themes(self):
        if not os.path.exists(self.file_path):
            return {"current_theme": "Default Dark", "themes": {}}
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {"current_theme": "Default Dark", "themes": {}}

    def save_themes(self):
        self.data["current_theme"] = self.current_theme_name
        self.data["themes"] = self.themes
        with open(self.file_path, 'w') as f:
            json.dump(self.data, f, indent=2)

    def get_current_theme_data(self):
        return self.themes.get(self.current_theme_name, self._get_default_dark_theme())

    def set_current_theme(self, theme_name):
        if theme_name in self.themes:
            self.current_theme_name = theme_name
            theme_data = self.get_current_theme_data()
            appearance_mode = "Dark" if "dark" in theme_name.lower() else "Light"
            ctk.set_appearance_mode(appearance_mode)
            self.save_themes()
            return True
        return False

    def get_theme_names(self):
        return sorted(list(self.themes.keys()))

    def save_theme(self, theme_name, theme_data):
        self.themes[theme_name] = theme_data
        self.save_themes()
        
    def delete_theme(self, theme_name):
        if theme_name in self.themes and "Default" not in theme_name:
            del self.themes[theme_name]
            if self.current_theme_name == theme_name:
                self.set_current_theme("Default Dark")
            self.save_themes()
            return True
        return False

    def _get_default_dark_theme(self):
        return {
          "colors": {
            "bg_primary": "#242424", "bg_secondary": "#2B2B2B", "bg_tertiary": "#323232",
            "text_primary": "#D4D4D4", "text_secondary": "#A0A0A0",
            "accent_primary": "#007ACC", "accent_hover": "#005F9E",
            "success": "#38761D", "error": "#990000", "warning": "#FFC700",
            "editor_code_bg": "#1E1E1E", "editor_code_text": "#D4D4D4",
            "editor_template_bg": "#2D2D30", "editor_template_text": "#CCCCCC",
            "editor_step_param_bg": "#3C3C3C"
          },
          "fonts": {
            "main_family": "SF Pro Text", "code_family": "SF Mono",
            "main_size": 13, "label_size": 14, "title_size": 20,
            "editor_code_family": "SF Mono", "editor_code_size": 13,
            "editor_template_family": "Georgia", "editor_template_size": 14,
            "editor_step_param_family": "SF Pro Text", "editor_step_param_size": 13
          },
          "styles": {
            "button_corner_radius": 6, "frame_border_width": 1
          }
        }
        
    def _get_default_light_theme(self):
        return {
          "colors": {
            "bg_primary": "#FFFFFF", "bg_secondary": "#F3F3F3", "bg_tertiary": "#EAEAEA",
            "text_primary": "#000000", "text_secondary": "#505050",
            "accent_primary": "#0078D7", "accent_hover": "#106EBE",
            "success": "#107C10", "error": "#A80000", "warning": "#FF8C00",
            "editor_code_bg": "#FFFFFF", "editor_code_text": "#000000",
            "editor_template_bg": "#F3F3F3", "editor_template_text": "#222222",
            "editor_step_param_bg": "#EAEAEA"
          },
          "fonts": {
            "main_family": "SF Pro Text", "code_family": "SF Mono",
            "main_size": 13, "label_size": 14, "title_size": 20,
            "editor_code_family": "SF Mono", "editor_code_size": 13,
            "editor_template_family": "Georgia", "editor_template_size": 14,
            "editor_step_param_family": "SF Pro Text", "editor_step_param_size": 13
          },
          "styles": {
            "button_corner_radius": 6, "frame_border_width": 1
          }
        }