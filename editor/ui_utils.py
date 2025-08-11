import customtkinter as ctk
from functools import reduce 
import operator
import re
import webbrowser

def get_nested(data, key_str):
    try:
        return reduce(operator.getitem, key_str.split('.'), data)
    except (KeyError, TypeError, AttributeError):
        return None

class ToolTip(ctk.CTkToplevel):
    def __init__(self, widget, text):
        super().__init__(widget)
        self.widget = widget
        self.text = text
        
        self.withdraw()
        self.overrideredirect(True)
        
        self.label = ctk.CTkLabel(self, text=self.text, corner_radius=5,
                                  fg_color=("#333333", "#444444"), text_color="white",
                                  wraplength=250, justify="left", padx=10, pady=5)
        self.label.pack()
        
        self.widget.bind("<Enter>", self.show)
        self.widget.bind("<Leave>", self.hide)
        self.widget.bind("<Button-1>", self.hide)

    def show(self, event=None):
        if not self.widget.winfo_exists(): return
        x = self.widget.winfo_rootx() + 20
        y = self.widget.winfo_rooty() + self.widget.winfo_height() + 5
        self.geometry(f"+{x}+{y}")
        self.deiconify()
        self.lift()

    def hide(self, event=None):
        self.withdraw()
        
    def update_text(self, new_text):
        self.text = new_text
        self.label.configure(text=self.text)

class ValidationMixin:
    """Mixin class providing common validation functionality for entry widgets"""
    
    def validate_entry_with_feedback(self, key, entry_widget, hint, available_vars=None):
        """
        Validates entry widget content based on hints and provides visual feedback
        
        Args:
            key: The parameter key being validated
            entry_widget: The CTkEntry widget to validate
            hint: Dictionary containing validation rules and hints
            available_vars: Set of available variable names for $variable validation
        
        Returns:
            bool: True if validation passes, False otherwise
        """
        is_valid = True
        error_message = ""
        value = entry_widget.get()
        rules = hint.get("validation", {})
        
        # Check if it's a variable reference and validate against available vars
        if value.startswith('$'):
            if available_vars is not None:
                found_vars = re.findall(r'\$(\w+)', value)
                if all(var in available_vars for var in found_vars):
                    entry_widget.configure(border_color=self.default_border_color)
                    if hasattr(self, 'tooltips') and key in self.tooltips:
                        self.tooltips[key].update_text(hint.get("tooltip", ""))
                    return True
                else:
                    is_valid = False
                    error_message = "Invalid variable reference(s)"
            else:
                entry_widget.configure(border_color=self.default_border_color)
                if hasattr(self, 'tooltips') and key in self.tooltips:
                    self.tooltips[key].update_text(hint.get("tooltip", ""))
                return True

        # Validate empty values
        if not value:
            if rules.get("type") in ["integer", "float"]:
                is_valid = False
                error_message = f"A valid {rules['type']} is required."
        else:
            # Type-specific validation
            try:
                if rules.get("type") == "integer":
                    num = int(value)
                    if "min_value" in rules and num < rules["min_value"]:
                        is_valid = False
                        error_message = f"Must be at least {rules['min_value']}."
                    if "max_value" in rules and num > rules["max_value"]:
                        is_valid = False
                        error_message = f"Must be no more than {rules['max_value']}."
                elif rules.get("type") == "float":
                    num = float(value)
                    if "min_value" in rules and num < rules["min_value"]:
                        is_valid = False
                        error_message = f"Must be at least {rules['min_value']}."
                    if "max_value" in rules and num > rules["max_value"]:
                        is_valid = False
                        error_message = f"Must be no more than {rules['max_value']}."
                if rules.get("type") == "string" and "regex" in rules:
                    if not re.match(rules["regex"], value):
                        is_valid = False
                        error_message = rules.get("regex_error", "Does not match required format.")
            except (ValueError, TypeError):
                is_valid = False
                error_message = "Invalid format."

        # Apply visual feedback
        entry_widget.configure(border_color="red" if not is_valid else self.default_border_color)
        
        # Update tooltip with error message if applicable
        if hasattr(self, 'tooltips') and key in self.tooltips:
            base_tooltip = hint.get("tooltip", "")
            full_tooltip = f"{base_tooltip}\n\n[ERROR] {error_message}" if not is_valid and error_message else base_tooltip
            self.tooltips[key].update_text(full_tooltip)
        
        return is_valid

    def apply_common_param_hints(self, key, entry_widget, hint):
        """
        Applies common parameter hints to an entry widget (tooltip, example, validation, doc link)
        
        Args:
            key: The parameter key
            entry_widget: The CTkEntry widget
            hint: Dictionary containing hint information
        """
        # Apply tooltip
        if "tooltip" in hint:
            if not hasattr(self, 'tooltips'):
                self.tooltips = {}
            self.tooltips[key] = ToolTip(entry_widget, hint["tooltip"])

        # Apply placeholder example
        if "example" in hint:
            entry_widget.configure(placeholder_text=str(hint['example']))

        # Apply validation with real-time feedback
        if "validation" in hint:
            available_vars = getattr(self, 'available_vars', None)
            entry_widget.bind("<KeyRelease>", 
                lambda event, k=key, e=entry_widget, h=hint: self.validate_entry_with_feedback(k, e, h, available_vars))
            self.validate_entry_with_feedback(key, entry_widget, hint, available_vars)

        # Apply documentation link
        if "doc_link" in hint:
            doc_btn = ctk.CTkButton(entry_widget.master, text="Docs", width=40, height=28, 
                                  command=lambda u=hint['doc_link']: webbrowser.open(u))
            doc_btn.pack(side="right", padx=(2,0))

class PresetSelectorMixin:
    """Mixin class providing preset selector widget functionality"""
    
    def create_preset_selector_widget(self, parent, key, value, hint, config_manager):
        """
        Creates a preset selector widget with dropdown and entry field
        
        Args:
            parent: Parent widget
            key: Parameter key
            value: Current value
            hint: Hint dictionary containing data_source path
            config_manager: Configuration manager to get preset data
            
        Returns:
            tuple: (option_menu, value_entry) widgets
        """
        # Get font settings
        param_font = getattr(self, '_get_param_font', lambda: ctk.CTkFont())()
        
        # Create main container
        widget_frame = ctk.CTkFrame(parent, fg_color="transparent")
        widget_frame.pack(side="left", fill="x", expand=True)

        # Get preset data
        data_source_path = hint.get("data_source")
        preset_data = get_nested(config_manager.config, data_source_path)
        if preset_data is None:
            print(f"DEBUG: GUI preset_selector could not find data_source '{data_source_path}' for parameter '{key}'.")
            preset_data = {}
        
        preset_names = ["Custom"] + sorted(preset_data.keys())
        
        # Create dropdown menu
        option_menu = ctk.CTkOptionMenu(widget_frame, values=preset_names, width=150, font=param_font)
        option_menu.pack(fill="x", expand=True)
        
        # Create value entry
        entry_bg_color = getattr(self, '_get_entry_bg_color', lambda: 'transparent')()
        value_entry = ctk.CTkEntry(widget_frame, font=param_font, fg_color=entry_bg_color)
        value_entry.insert(0, value)
        value_entry.pack(fill="x", expand=True, pady=(2, 0))
        
        # Store entry reference for form processing
        if hasattr(self, 'param_entries'):
            self.param_entries[f'param_val_{key}'] = value_entry
        elif hasattr(self, 'input_entries'):
            self.input_entries[key] = ctk.StringVar()
            self.input_entries[key].set(value)
            value_entry.configure(textvariable=self.input_entries[key])

        # Setup interactions
        def on_menu_select(choice):
            if choice != "Custom":
                selected_value = preset_data.get(choice, "")
                value_entry.delete(0, "end")
                value_entry.insert(0, selected_value)
                if hasattr(self, 'input_entries') and key in self.input_entries:
                    self.input_entries[key].set(selected_value)
            # Trigger validation if available
            if hasattr(self, 'validate_entry_variables'):
                self.validate_entry_variables(value_entry)
        
        def on_entry_change(*args):
            current_text = value_entry.get()
            # Update variable validation if available
            if hasattr(self, 'validate_entry_variables'):
                self.validate_entry_variables(value_entry)
            # Update dropdown selection if not a variable reference
            if not current_text.startswith('$'):
                matching_preset = "Custom"
                for name, p_value in preset_data.items():
                    if current_text == p_value:
                        matching_preset = name
                        break
                option_menu.set(matching_preset)

        # Connect event handlers
        option_menu.configure(command=on_menu_select)
        if hasattr(self, 'input_entries') and key in self.input_entries:
            self.input_entries[key].trace_add("write", on_entry_change)
        else:
            value_entry.bind("<KeyRelease>", on_entry_change)
        
        # Apply hints
        if hasattr(self, 'apply_common_param_hints'):
            self.apply_common_param_hints(key, value_entry, hint)
        
        # Set initial dropdown state
        on_entry_change()
        
        return option_menu, value_entry

class ThemeMixin:
    """Mixin class providing theme-related utility methods"""
    
    def get_theme_color(self, key, default_color):
        """Get theme color with fallback to default"""
        if hasattr(self, 'theme') and self.theme:
            return self.theme.get("colors", {}).get(key, default_color)
        return default_color

    def get_theme_font(self, key, default_font):
        """Get theme font with fallback to default"""
        if hasattr(self, 'theme') and self.theme:
            font_info = self.theme.get("fonts", {})
            family = font_info.get(f"{key}_family", default_font[0])
            size = font_info.get(f"{key}_size", default_font[1])
            return (family, size)
        return default_font