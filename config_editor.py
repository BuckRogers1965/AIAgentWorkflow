import pygame, json, sys, os
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token
from pygments.styles import get_style_by_name

# --- Constants and Initialization (Unchanged) ---
CONFIG_PATH = "config.json"
WIDTH, HEIGHT = 1400, 800
WHITE, BLACK, GRAY, BLUE, GREEN, RED, DARK_GRAY, LIGHT_BLUE = (255,255,255), (0,0,0), (180,180,180), (0,120,255), (0,180,0), (180,0,0), (100,100,100), (200,220,255)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Workflow Editor")
font = pygame.font.SysFont(None, 22)
small_font = pygame.font.SysFont(None, 18)
code_font = pygame.font.SysFont("monospace", 16)
clock = pygame.time.Clock()

style = get_style_by_name('default')
PYGMENTS_COLORS = {
    Token.Keyword: (0, 0, 200), Token.Name.Function: (0, 100, 100),
    Token.Name.Class: (0, 100, 100), Token.String: (0, 150, 0),
    Token.Number: (150, 50, 0), Token.Comment: (128, 128, 128),
    Token.Operator: (150, 0, 150),
}

def get_pygments_color(token):
    while token not in PYGMENTS_COLORS and token.parent: token = token.parent
    return PYGMENTS_COLORS.get(token, BLACK)

# --- CodeEditor Class (Unchanged) ---
class CodeEditor:
    def __init__(self, rect, initial_text=""):
        self.rect = rect; self.lines = initial_text.splitlines() or [""]
        self.cursor_line, self.cursor_col = 0, 0; self.scroll_y, self.scroll_x = 0, 0
        self.line_height = code_font.get_height() + 2; self.margin_left = 45
    def get_value(self): return "\n".join(self.lines)
    def get_content_size(self):
        total_height = len(self.lines) * self.line_height
        max_width = max(code_font.size(line.replace('\t', '    '))[0] for line in self.lines) if self.lines else 0
        return max_width + self.margin_left, total_height
    def _ensure_cursor_visible(self):
        content_w, content_h = self.get_content_size()
        visible_lines = self.rect.height // self.line_height
        if self.cursor_line < self.scroll_y: self.scroll_y = self.cursor_line
        if self.cursor_line >= self.scroll_y + visible_lines: self.scroll_y = self.cursor_line - visible_lines + 1
        self.scroll_y = max(0, min(self.scroll_y, (content_h - self.rect.height) / self.line_height if content_h > self.rect.height else 0))
        cursor_x_offset = code_font.size(self.lines[self.cursor_line][:self.cursor_col].replace('\t', '    '))[0]
        if cursor_x_offset < self.scroll_x: self.scroll_x = cursor_x_offset - 20
        if cursor_x_offset > self.scroll_x + self.rect.width - self.margin_left - 20: self.scroll_x = cursor_x_offset - (self.rect.width - self.margin_left - 20)
        self.scroll_x = max(0, min(self.scroll_x, content_w - self.rect.width if content_w > self.rect.width else 0))
    def draw(self):
        pygame.draw.rect(screen, WHITE, self.rect); pygame.draw.rect(screen, BLACK, self.rect, 1)
        original_clip = screen.get_clip(); screen.set_clip(self.rect.inflate(-2, -2))
        visible_lines = self.rect.height // self.line_height
        for i in range(visible_lines):
            line_idx = int(self.scroll_y) + i
            if line_idx >= len(self.lines): break
            num_surf = code_font.render(str(line_idx + 1), True, GRAY); screen.blit(num_surf, (self.rect.x + 5, self.rect.y + i * self.line_height))
            line_text = self.lines[line_idx]; x_pos = self.rect.x + self.margin_left - self.scroll_x
            for token, text in lex(line_text, PythonLexer()):
                text_surf = code_font.render(text.replace('\t', '    '), True, get_pygments_color(token)); screen.blit(text_surf, (x_pos, self.rect.y + i * self.line_height)); x_pos += text_surf.get_width()
        if pygame.time.get_ticks() % 1000 < 500 and int(self.scroll_y) <= self.cursor_line < int(self.scroll_y) + visible_lines:
            cursor_x = self.rect.x + self.margin_left - self.scroll_x + code_font.size(self.lines[self.cursor_line][:self.cursor_col].replace('\t', '    '))[0]; cursor_y = self.rect.y + (self.cursor_line - self.scroll_y) * self.line_height
            pygame.draw.line(screen, BLACK, (cursor_x, cursor_y), (cursor_x, cursor_y + self.line_height), 1)
        screen.set_clip(original_clip); return self.draw_scrollbars()
    def draw_scrollbars(self):
        content_w, content_h = self.get_content_size(); v_thumb_rect, h_thumb_rect = None, None
        if content_h > self.rect.height:
            bar_area_h = self.rect.height - 12; thumb_h = max(20, bar_area_h * self.rect.height / content_h); thumb_y = self.rect.y + (self.scroll_y * self.line_height / (content_h - self.rect.height)) * (bar_area_h - thumb_h)
            pygame.draw.rect(screen, GRAY, (self.rect.right - 12, self.rect.y, 10, self.rect.height)); v_thumb_rect = pygame.Rect(self.rect.right - 12, thumb_y, 10, thumb_h); pygame.draw.rect(screen, DARK_GRAY, v_thumb_rect)
        if content_w > self.rect.width:
            bar_area_w = self.rect.width - 12; thumb_w = max(20, bar_area_w * self.rect.width / content_w); thumb_x = self.rect.x + (self.scroll_x / (content_w - self.rect.width)) * (bar_area_w - thumb_w)
            pygame.draw.rect(screen, GRAY, (self.rect.x, self.rect.bottom - 12, self.rect.width, 10)); h_thumb_rect = pygame.Rect(thumb_x, self.rect.bottom - 12, thumb_w, 10); pygame.draw.rect(screen, DARK_GRAY, h_thumb_rect)
        return v_thumb_rect, h_thumb_rect
    def handle_click(self, event):
        mx, my = event.pos; self.cursor_line = max(0, min(len(self.lines) - 1, int(self.scroll_y) + (my - self.rect.y) // self.line_height)); line_text = self.lines[self.cursor_line].replace('\t', '    '); click_x = mx - (self.rect.x + self.margin_left) + self.scroll_x; min_dist, best_col = float('inf'), 0
        for i in range(len(line_text) + 1):
            dist = abs(click_x - code_font.size(line_text[:i])[0]);
            if dist < min_dist: min_dist, best_col = dist, i
        self.cursor_col = best_col; self._ensure_cursor_visible()
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN:
            current_line = self.lines[self.cursor_line]
            if event.key == pygame.K_BACKSPACE:
                if self.cursor_col > 0: self.lines[self.cursor_line] = current_line[:self.cursor_col-1] + current_line[self.cursor_col:]; self.cursor_col -= 1
                elif self.cursor_line > 0: prev_line_len = len(self.lines[self.cursor_line - 1]); self.lines[self.cursor_line-1] += current_line; self.lines.pop(self.cursor_line); self.cursor_line -= 1; self.cursor_col = prev_line_len
            elif event.key == pygame.K_DELETE:
                if self.cursor_col < len(current_line): self.lines[self.cursor_line] = current_line[:self.cursor_col] + current_line[self.cursor_col+1:]
                elif self.cursor_line < len(self.lines) - 1: self.lines[self.cursor_line] += self.lines.pop(self.cursor_line + 1)
            elif event.key in [pygame.K_RETURN, pygame.K_KP_ENTER]:
                leading_whitespace = "".join(char for char in current_line if char.isspace()) if current_line.lstrip() == '' else "".join(current_line.split(current_line.lstrip())[0]); remainder = current_line[self.cursor_col:]
                self.lines[self.cursor_line] = current_line[:self.cursor_col]; self.lines.insert(self.cursor_line + 1, leading_whitespace + remainder); self.cursor_line += 1; self.cursor_col = len(leading_whitespace)
            elif event.key == pygame.K_LEFT:
                if self.cursor_col > 0: self.cursor_col -= 1
                elif self.cursor_line > 0: self.cursor_line -= 1; self.cursor_col = len(self.lines[self.cursor_line])
            elif event.key == pygame.K_RIGHT:
                if self.cursor_col < len(current_line): self.cursor_col += 1
                elif self.cursor_line < len(self.lines) - 1: self.cursor_line += 1; self.cursor_col = 0
            elif event.key == pygame.K_UP:
                if self.cursor_line > 0: self.cursor_line -= 1; self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_line]))
            elif event.key == pygame.K_DOWN:
                if self.cursor_line < len(self.lines) - 1: self.cursor_line += 1; self.cursor_col = min(self.cursor_col, len(self.lines[self.cursor_line]))
            elif event.key == pygame.K_TAB: self.lines[self.cursor_line] = current_line[:self.cursor_col] + '    ' + current_line[self.cursor_col:]; self.cursor_col += 4
            elif event.unicode: self.lines[self.cursor_line] = current_line[:self.cursor_col] + event.unicode + current_line[self.cursor_col:]; self.cursor_col += len(event.unicode)
            self._ensure_cursor_visible()
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button in [4, 5]:
            scroll_amount = -1 if event.button == 4 else 1; mods = pygame.key.get_mods(); content_w, content_h = self.get_content_size()
            if mods & pygame.KMOD_SHIFT: self.scroll_x = max(0, min(self.scroll_x + scroll_amount * 30, content_w - self.rect.width if content_w > self.rect.width else 0))
            else: self.scroll_y = max(0, min(self.scroll_y + scroll_amount, len(self.lines) - 1 if content_h > self.rect.height else 0))

# --- Utility Functions (Unchanged) ---
def load_config():
    if not os.path.exists(CONFIG_PATH): return {"agents": {}}
    with open(CONFIG_PATH, 'r') as f: return json.load(f)
def save_config(cfg):
    if "agents" in cfg: cfg["agents"] = {k: cfg["agents"][k] for k in sorted(cfg["agents"].keys())}
    with open(CONFIG_PATH, 'w') as f: json.dump(cfg, f, indent=2)
def draw_text(txt, x, y, color=BLACK, font_obj=None, surface=None):
    if font_obj is None: font_obj = font
    if surface is None: surface = screen
    surface.blit(font_obj.render(str(txt), True, color), (x, y))
def button(txt, x, y, w=250, h=30, color=GRAY):
    r = pygame.Rect(x, y, w, h); pygame.draw.rect(screen, color, r); draw_text(txt, x + 6, y + 6)
    return r
def small_button(txt, x, y, w=60, h=20, color=GRAY, surface=None):
    if surface is None: surface = screen
    r = pygame.Rect(x, y, w, h); pygame.draw.rect(surface, color, r); draw_text(txt, x + 4, y + 2, BLACK, small_font, surface=surface)
    return r

# --- Side Panel Class (FIXED) ---
class StepSidePanel:
    def __init__(self, step_index, step_data, all_agents_data):
        self.rect = pygame.Rect(1080, 0, 320, HEIGHT)
        self.index = step_index
        self.step_data = json.loads(json.dumps(step_data))
        self.all_agents_data = all_agents_data
        self.scroll_y = 0; self.content_height = 0; self.thumb_rect = None
        self.active_field = None; self.cursor_positions = {}
    
    def reset_state(self):
        """Reset the panel state when switching between steps"""
        self.active_field = None
        self.cursor_positions = {}
        self.scroll_y = 0
    
    def get_updated_data(self): return self.step_data
    def is_active(self): return self.active_field is not None
    
    def handle_event(self, event):
        if event.type == pygame.KEYDOWN and self.active_field: self._handle_typing(event)
        elif event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            if event.button == 4: self.scroll_y = max(0, self.scroll_y - 30)
            if event.button == 5: self.scroll_y = max(0, self.scroll_y + 30)
    
    def _handle_typing(self, e):
        if not self.active_field: return
        field_type, field_key = self.active_field
        text, is_key_edit = "", False
        if field_type == 'agent': text = self.step_data.get('agent', '')
        elif field_type == 'output': text = self.step_data.get('output', [])[field_key]
        elif field_type == 'param_key': text, is_key_edit = field_key, True
        elif field_type == 'param_value': text = self.step_data.get('params', {}).get(field_key, '')
        cursor_pos = self.cursor_positions.get(self.active_field, 0); new_text = str(text)
        if e.key == pygame.K_BACKSPACE:
            if cursor_pos > 0: new_text = new_text[:cursor_pos-1] + new_text[cursor_pos:]; cursor_pos -= 1
        elif e.key == pygame.K_DELETE: new_text = new_text[:cursor_pos] + new_text[cursor_pos+1:]
        elif e.key == pygame.K_LEFT: cursor_pos = max(0, cursor_pos-1)
        elif e.key == pygame.K_RIGHT: cursor_pos = min(len(new_text), cursor_pos+1)
        elif e.key == pygame.K_ESCAPE: self.active_field = None; return
        elif e.unicode: new_text = new_text[:cursor_pos] + e.unicode + new_text[cursor_pos:]; cursor_pos += len(e.unicode)
        self.cursor_positions[self.active_field] = cursor_pos
        if is_key_edit:
            old_key = field_key
            if old_key in self.step_data['params'] and old_key != new_text and new_text not in self.step_data['params']:
                self.step_data['params'][new_text] = self.step_data['params'].pop(old_key)
                new_active_field = ('param_key', new_text)
                self.cursor_positions[new_active_field] = self.cursor_positions.pop(self.active_field, cursor_pos)
                self.active_field = new_active_field
        else:
            if field_type == 'agent': self.step_data['agent'] = new_text
            elif field_type == 'output': self.step_data['output'][field_key] = new_text
            elif field_type == 'param_value' and field_key in self.step_data['params']: self.step_data['params'][field_key] = new_text
    
    def _text_input(self, surface, event, field_id, value, x, y, w=300, h=30):
        r = pygame.Rect(x, y, w, h); is_active = (self.active_field == field_id)
        pygame.draw.rect(surface, WHITE, r); pygame.draw.rect(surface, LIGHT_BLUE if is_active else BLACK, r, 2 if is_active else 1)
        draw_text(value, x + 6, y + 6, BLACK, font, surface)
        if is_active and pygame.time.get_ticks() % 1000 < 500:
            cursor_pos = self.cursor_positions.get(field_id, 0); cursor_x = x + 6 + font.size(str(value)[:cursor_pos])[0]
            pygame.draw.line(surface, BLACK, (cursor_x, y + 4), (cursor_x, y + 26), 1)
        if event and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and r.collidepoint(event.pos):
            self.active_field = field_id; click_x = event.pos[0] - (x + 6); min_dist, best_col = float('inf'), 0
            for i in range(len(str(value)) + 1):
                dist = abs(click_x - font.size(str(value)[:i])[0])
                if dist < min_dist: min_dist, best_col = dist, i
            self.cursor_positions[field_id] = best_col
        return r.height
    
    def _draw_list_editor(self, surface, event, title, data_list, x, y, w):
        draw_text(f"{title}:", x, y, font_obj=font, surface=surface); y += 25; item_to_remove = -1
        for i, item in enumerate(data_list):
            h = self._text_input(surface, event, (title, i), item, x + 20, y, w - 60)
            remove_btn_rect = pygame.Rect(x + w - 35, y + (h - 20)//2, 20, 20)
            small_button("X", remove_btn_rect.x, remove_btn_rect.y, 20, 20, RED, surface)
            if event and event.type == pygame.MOUSEBUTTONDOWN and remove_btn_rect.collidepoint(event.pos): item_to_remove = i
            y += h + 5
        if item_to_remove != -1: data_list.pop(item_to_remove)
        add_btn_rect = pygame.Rect(x + 20, y, 60, 25)
        small_button("+ Add", add_btn_rect.x, add_btn_rect.y, 60, 25, GREEN, surface)
        if event and event.type == pygame.MOUSEBUTTONDOWN and add_btn_rect.collidepoint(event.pos): data_list.append("new_item")
        return y + 35
    
    def _draw_dict_editor(self, surface, event, title, data_dict, agent_info, x, y, w):
        draw_text(f"{title}:", x, y, font_obj=font, surface=surface); y += 25
        required_inputs = set(agent_info.get("inputs", [])) if agent_info else set(); keys_to_remove = []
        for k in list(data_dict.keys()):
            v = data_dict[k]
            key_h = self._text_input(surface, event, ('param_key', k), k, x + 20, y, 100)
            val_h = self._text_input(surface, event, ('param_value', k), v, x + 130, y, w - 180)
            row_height = max(key_h, val_h)
            if not (k in required_inputs) or not agent_info:
                remove_btn_rect = pygame.Rect(x + w - 35, y + (row_height-20)//2, 20, 20)
                small_button("X", remove_btn_rect.x, remove_btn_rect.y, 20, 20, RED, surface)
                if event and event.type == pygame.MOUSEBUTTONDOWN and remove_btn_rect.collidepoint(event.pos): keys_to_remove.append(k)
            y += row_height + 5
        for k in keys_to_remove: data_dict.pop(k, None)
        add_btn_rect = pygame.Rect(x + 20, y, 60, 25)
        small_button("+ Add", add_btn_rect.x, add_btn_rect.y, 60, 25, GREEN, surface)
        if event and event.type == pygame.MOUSEBUTTONDOWN and add_btn_rect.collidepoint(event.pos): data_dict[f"param_{len(data_dict)}"] = "value"
        return y + 35
    
    def draw(self, main_surface, event):
        pygame.draw.rect(main_surface, (245, 245, 245), self.rect)
        panel_surface = pygame.Surface((self.rect.width, 2000)); panel_surface.fill((245, 245, 245))
        
        # FIXED: Only create relative_event if the event is within our rect and not on scrollbar
        relative_event = None
        if event and event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos):
            # Don't create relative event if clicking on scrollbar
            if not (self.thumb_rect and self.thumb_rect.collidepoint(event.pos)):
                relative_event = pygame.event.Event(event.type, {
                    'pos': (event.pos[0] - self.rect.x, event.pos[1] - self.rect.y + self.scroll_y), 
                    'button': event.button
                })
        
        content_y, content_x, content_w = 10, 10, self.rect.width - 20
        draw_text(f"Step {self.index} Properties:", content_x, content_y, surface=panel_surface); content_y += 30
        draw_text("Agent:", content_x, content_y, surface=panel_surface)
        content_y += self._text_input(panel_surface, relative_event, ('agent', None), self.step_data.get('agent', ''), content_x, content_y + 20) + 10
        content_y = self._draw_list_editor(panel_surface, relative_event, 'output', self.step_data.setdefault('output', []), content_x, content_y, content_w)
        agent_info = self.all_agents_data.get(self.step_data.get("agent"), {})
        content_y = self._draw_dict_editor(panel_surface, relative_event, 'params', self.step_data.setdefault('params', {}), agent_info, content_x, content_y, content_w)
        if agent_info:
            draw_text("--- Referenced Agent Info ---", content_x, content_y, DARK_GRAY, surface=panel_surface); content_y += 25
            for k_info, v_info in agent_info.items():
                if isinstance(v_info, list) and v_info:
                    draw_text(f"{k_info}:", content_x + 10, content_y, DARK_GRAY, surface=panel_surface); content_y += 20
                    for item in v_info: draw_text(f"  • {item}", content_x + 20, content_y, DARK_GRAY, surface=panel_surface); content_y += 18
                    content_y += 5
        self.content_height = content_y
        main_surface.blit(panel_surface, self.rect.topleft, (0, self.scroll_y, self.rect.width, self.rect.height))
        if self.content_height > self.rect.height:
            thumb_h = max(20, self.rect.height * self.rect.height / self.content_height)
            thumb_y = self.rect.y + (self.scroll_y / (self.content_height - self.rect.height)) * (self.rect.height - thumb_h)
            self.thumb_rect = pygame.Rect(self.rect.right - 12, thumb_y, 10, thumb_h)
            pygame.draw.rect(main_surface, GRAY, (self.rect.right - 12, self.rect.y, 10, self.rect.height)); pygame.draw.rect(main_surface, DARK_GRAY, self.thumb_rect)
        else: self.thumb_rect = None
        pygame.draw.rect(main_surface, BLACK, self.rect, 1)

# --- Editor Class (FIXED) ---
class Editor:
    ALLOWED_AGENT_TYPES = {"workflow", "template", "proc"}
    def __init__(self):
        self.cfg = load_config(); self.agents = self.cfg.get("agents", {})
        self.selected, self.editing = None, None
        self.agent_scroll, self.editor_scroll = 0, 0
        self.dragging_step, self.drop_target, self.active_idx = None, None, None
        self.field_edit, self.cursor_positions, self.code_editor = {}, {}, None
        self.dragging_scrollbar = None
        self.agent_list_thumb, self.editor_panel_thumb = None, None
        self.code_editor_v_thumb, self.code_editor_h_thumb = None, None
        self.side_panel = None

    def open_side_panel(self, index):
        """Cleanly opens a side panel for a given index."""
        if self.editing and 0 <= index < len(self.editing.get("steps", [])):
            # FIXED: Always close existing panel first to save state
            if self.side_panel:
                self.close_side_panel()
            
            self.active_idx = index
            step_data = self.editing["steps"][self.active_idx]
            self.side_panel = StepSidePanel(self.active_idx, step_data, self.agents)
            # FIXED: Reset the panel state to ensure clean initialization
            self.side_panel.reset_state()
        else:
            self.active_idx = None
            self.side_panel = None

    def close_side_panel(self):
        """Cleanly saves and closes any active side panel."""
        if self.side_panel and self.active_idx is not None:
            # Save the data back to the step
            self.editing["steps"][self.active_idx] = self.side_panel.get_updated_data()
        
        # FIXED: Always clear these regardless of state
        self.side_panel = None
        self.active_idx = None

    # --- All original functionality methods are preserved below ---
    def generate_unique_name(self, base):
        i = 1;
        while f"{base}_{i}" in self.agents: i += 1
        return f"{base}_{i}"
    def add_agent(self, agent_type):
        name = self.generate_unique_name(f"new_{agent_type}")
        new_agent = {"type": agent_type, "help": "", "inputs": [], "optional_inputs": [], "outputs": []}
        if agent_type == "workflow": new_agent.update({"return_on_fail": 0, "steps": []})
        elif agent_type == "template": new_agent.update({"prompt": ""})
        elif agent_type == "proc": new_agent.update({"function": "", "function_def": ""})
        self.agents[name] = new_agent
        self.selected, self.editing = name, json.loads(json.dumps(self.agents[name]))
        self.active_idx, self.field_edit, self.code_editor, self.side_panel = None, {}, None, None
    def add_agent_as_step(self, agent_name):
        if self.editing and self.editing.get("type") == "workflow":
            agent_info = self.agents.get(agent_name, {})
            new_params = {key: f"${key}" for key in agent_info.get("inputs", [])}
            new_step = {"agent": agent_name, "params": new_params, "output": agent_info.get("outputs", ["output"]).copy()}
            self.editing["steps"].append(new_step)
    def remove_workflow_step(self, idx):
        if self.editing and self.editing.get("type") == "workflow" and 0 <= idx < len(self.editing["steps"]):
            if self.active_idx == idx: self.close_side_panel()
            self.editing["steps"].pop(idx)
            if self.active_idx and self.active_idx > idx: self.active_idx -= 1
    def text_input(self, key, val, event, x, y, w=300, min_lines=1):
        if self.side_panel and self.side_panel.is_active(): self.field_edit['active'] = None
        if key not in self.field_edit: self.field_edit[key] = str(val)
        text = self.field_edit[key]; words = text.split(' '); lines = [""]
        for word in words:
            if "\n" in word:
                parts = word.split("\n")
                for i, part in enumerate(parts):
                    if font.size(lines[-1] + ' ' + part)[0] <= w - 12: lines[-1] += (' ' if lines[-1] else '') + part
                    else: lines.append(part)
                    if i < len(parts) - 1: lines.append("")
            else:
                test_line = lines[-1] + (' ' if lines[-1] else '') + word
                if font.size(test_line)[0] <= w-12: lines[-1] = test_line
                else: lines.append(word)
        while len(lines) < min_lines: lines.append("")
        line_height = 22; h = max(30, len(lines) * line_height + 8); r = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, WHITE, r); pygame.draw.rect(screen, BLACK, r, 1)
        is_active = self.field_edit.get("active") == key
        if is_active: pygame.draw.rect(screen, LIGHT_BLUE, r, 2)
        for i, line in enumerate(lines): draw_text(line, x + 6, y + 6 + i * line_height)
        if is_active:
            cursor_pos = self.cursor_positions.get(key, len(text)); char_count, cursor_x, cursor_y = 0, x + 6, y + 6; found = False
            for i, line in enumerate(lines):
                if char_count + len(line)+1 >= cursor_pos:
                    cursor_x = x + 6 + font.size(line[:cursor_pos - char_count])[0]; cursor_y = y + 6 + i * line_height; found = True; break
                char_count += len(line) + 1
            if not found: cursor_x, cursor_y = x + 6 + font.size(lines[-1])[0], y + 6 + (len(lines)-1) * line_height
            if pygame.time.get_ticks() % 1000 < 500: pygame.draw.line(screen, BLACK, (cursor_x, cursor_y), (cursor_x, cursor_y + 18), 1)
        if event and event.type == pygame.MOUSEBUTTONDOWN and r.collidepoint(event.pos):
            self.field_edit["active"] = key; self.code_editor = None
            if self.side_panel: self.side_panel.active_field = None
            mx, my = event.pos; clicked_line_idx = max(0, min(len(lines) - 1, (my - y - 6) // line_height))
            line_text = lines[clicked_line_idx]; char_pos = 0
            for i in range(len(line_text) + 1):
                char_x = x + 6 + font.size(line_text[:i])[0]; next_char_width = font.size(line_text[i:i+1])[0] if i < len(line_text) else 0
                if mx <= char_x + next_char_width / 2: char_pos = i; break
            abs_pos = sum(len(l) + 1 for l in lines[:clicked_line_idx]) + char_pos; self.cursor_positions[key] = min(abs_pos, len(text))
        return self.field_edit.get(key,str(val)), h
    def draw_list_editor(self, key, lst, event, x, y, max_width=400):
        draw_text(f"{key}:", x, y); y += 25; item_to_remove = -1
        for i, item in enumerate(lst):
            new_val, h = self.text_input(f"{key}_{i}", item, event, x + 20, y, min(250, max_width - 100))
            if new_val != item: lst[i] = new_val
            remove_btn = small_button("X", x + min(280, max_width - 80), y + (h - 20) // 2, 20, 20, RED)
            if event and event.type == pygame.MOUSEBUTTONDOWN and remove_btn.collidepoint(event.pos): item_to_remove = i
            y += h + 5
        if item_to_remove != -1: lst.pop(item_to_remove)
        add_btn = small_button("+ Add", x + 20, y, 60, 25, GREEN)
        if event and event.type == pygame.MOUSEBUTTONDOWN and add_btn.collidepoint(event.pos): lst.append("new_item")
        return y + 35
    def draw_dict_editor(self, key, dct, event, x, y, max_width=400, agent_info=None):
        draw_text(f"{key}:", x, y); y += 25
        required_inputs = set(agent_info.get("inputs", [])) if agent_info else set()
        keys_to_remove = []; key_updates = {}
        for k, v in list(dct.items()):
            key_width, val_width = min(120, max_width // 3), min(200, max_width - (max_width // 3) - 60)
            new_key, key_h = self.text_input(f"{key}_key_{k}", k, event, x + 20, y, key_width)
            new_val, val_h = self.text_input(f"{key}_val_{k}", v, event, x + 30 + key_width, y, val_width)
            row_height = max(key_h, val_h)
            if new_key != k: key_updates[k] = new_key
            dct[k] = new_val
            if not (k in required_inputs) or not agent_info:
                remove_btn = small_button("X", x + max_width - 40, y + (row_height - 20) // 2, 20, 20, RED)
                if event and event.type == pygame.MOUSEBUTTONDOWN and remove_btn.collidepoint(event.pos): keys_to_remove.append(k)
            y += row_height + 5
        for old_k, new_k in key_updates.items():
            if new_k not in dct: dct[new_k] = dct.pop(old_k)
        for k in keys_to_remove: dct.pop(k, None)
        add_btn = small_button("+ Add", x + 20, y, 60, 25, GREEN)
        if event and event.type == pygame.MOUSEBUTTONDOWN and add_btn.collidepoint(event.pos): dct[f"param_{len(dct)}"] = "value"
        return y + 35
    def draw_checkbox(self, value, event, x, y):
        checkbox_rect = pygame.Rect(x, y, 16, 16); pygame.draw.rect(screen, WHITE, checkbox_rect); pygame.draw.rect(screen, BLACK, checkbox_rect, 1)
        if value: pygame.draw.lines(screen, BLACK, False, [(x+3,y+8),(x+7,y+12),(x+13,y+4)], 2)
        if event and event.type == pygame.MOUSEBUTTONDOWN and checkbox_rect.collidepoint(event.pos): return not value
        return value
    def draw_scrollbar(self, scroll, total_h, x, y_offset=0, area_h=HEIGHT):
        if total_h <= area_h: return None
        thumb_h = max(20, area_h * area_h / total_h); thumb_y = y_offset + (scroll / max(1, total_h - area_h)) * (area_h - thumb_h)
        thumb_rect = pygame.Rect(x, thumb_y, 10, thumb_h); pygame.draw.rect(screen, GRAY, (x, y_offset, 10, area_h)); pygame.draw.rect(screen, DARK_GRAY, thumb_rect)
        return thumb_rect
    def draw_agent_list(self, event):
        start_y, button_h, spacing = 20, 30, 10; draw_text("Agents:", 20, 0)
        add_buttons = [("Add Workflow", "workflow"), ("Add Template", "template"), ("Add Proc", "proc")]
        for i, (label, agent_type) in enumerate(add_buttons):
            r = button(label, 20, start_y + i * (button_h + spacing), 250, button_h)
            if event and event.type == pygame.MOUSEBUTTONDOWN and r.collidepoint(event.pos): self.add_agent(agent_type)
        agent_y_start = start_y + len(add_buttons) * (button_h + spacing) + spacing; view_height = HEIGHT - agent_y_start - 10
        y = agent_y_start - self.agent_scroll
        if self.editing and self.editing.get("type") == "workflow": draw_text("Click agent to add as step:", 20, agent_y_start - 20, BLUE, small_font)
        clip_rect = pygame.Rect(20, agent_y_start, 260, view_height); screen.set_clip(clip_rect)
        sorted_keys = sorted(self.agents.keys())
        for k in sorted_keys:
            agent_type = self.agents.get(k, {}).get("type"); is_valid_type = agent_type in self.ALLOWED_AGENT_TYPES
            color = BLUE if k == self.selected else (GRAY if is_valid_type else DARK_GRAY)
            r = button(f"[{agent_type[0] if agent_type else '?'}] {k}", 20, y, 220, 30, color)
            delete_btn = pygame.Rect(245, y + 5, 20, 20); pygame.draw.rect(screen, RED, delete_btn); draw_text("X", 250, y + 8, WHITE, small_font)
            if event and event.type == pygame.MOUSEBUTTONDOWN:
                if r.collidepoint(event.pos) and is_valid_type:
                    if self.editing and self.editing.get("type") == "workflow": self.add_agent_as_step(k)
                    else: self.close_side_panel(); self.selected, self.editing = k, json.loads(json.dumps(self.agents[k])); self.field_edit, self.active_idx, self.code_editor = {}, None, None
                elif delete_btn.collidepoint(event.pos) and k != self.selected: self.close_side_panel(); del self.agents[k]; self.cfg["agents"] = self.agents; save_config(self.cfg); break
            y += 35
        screen.set_clip(None); self.agent_list_thumb = self.draw_scrollbar(self.agent_scroll, len(self.agents) * 35, 290, y_offset=agent_y_start, area_h=view_height)
    def draw_workflow(self, event):
        middle_panel_width = 720; middle_panel = pygame.Rect(340, 0, middle_panel_width, HEIGHT); pygame.draw.rect(screen, (250, 250, 250), middle_panel); screen.set_clip(middle_panel)
        content_y = 20 - self.editor_scroll; draw_text("Agent Name:", 340, content_y)
        new_name, name_h = self.text_input("agent_name", self.selected, event, 460, content_y-5, 400); self.field_edit["agent_name"] = new_name; content_y += name_h + 10
        draw_text("Workflow Properties:", 340, content_y); content_y += 30; draw_text("help:", 340, content_y)
        val, h = self.text_input("wf_help", self.editing.get("help", ""), event, 460, content_y, 500); self.editing["help"] = val; content_y += h + 10
        draw_text("return_on_fail:", 340, content_y); self.editing["return_on_fail"] = self.draw_checkbox(self.editing.get("return_on_fail", 0), event, 460, content_y); content_y += 30
        for key in ["inputs", "optional_inputs", "outputs"]: content_y = self.draw_list_editor(f"wf_{key}", self.editing.setdefault(key, []), event, 340, content_y, middle_panel_width - 20)
        content_y += 20; draw_text("Workflow Steps: (A bug requires you to close a step before opening a new step.)", 340, content_y); content_y += 40
        step_to_remove = -1; steps = self.editing.get("steps", [])
        for i, step in enumerate(steps):
            r = pygame.Rect(340, content_y, middle_panel_width - 40, 30); pygame.draw.rect(screen, BLUE if i == self.active_idx else GRAY, r)
            draw_text(f"{i}. {step.get('agent', 'unknown')}", 346, content_y + 6)
            remove_btn = pygame.Rect(340 + middle_panel_width - 65, content_y + 5, 20, 20); pygame.draw.rect(screen, RED, remove_btn); draw_text("X", 340 + middle_panel_width - 60, content_y + 8, WHITE, small_font)
            
            # FIXED: This is the new, robust logic block for handling clicks.
            if event and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if r.collidepoint(event.pos):
                    # FIXED: Clear any active main editor fields when clicking step
                    self.field_edit['active'] = None
                    if self.code_editor:
                        self.code_editor = None
                    
                    old_idx = self.active_idx
                    if i == old_idx:
                        # Clicking the same step - toggle close
                        self.close_side_panel()
                    else:
                        # Clicking different step - switch to it
                        self.open_side_panel(i)
                elif remove_btn.collidepoint(event.pos):
                    step_to_remove = i

            if self.dragging_step is not None and r.collidepoint(pygame.mouse.get_pos()): self.drop_target = i
            content_y += 40
        if step_to_remove != -1: self.remove_workflow_step(step_to_remove)
        screen.set_clip(None); self.editor_panel_thumb = self.draw_scrollbar(self.editor_scroll, content_y, 340 + middle_panel_width - 10, 0, middle_panel.height)
    def draw_agent_fields(self, event):
        agent_type = self.editing['type']; middle_panel_width = 1040 if agent_type == 'proc' else 720; middle_panel = pygame.Rect(340, 0, middle_panel_width, HEIGHT); pygame.draw.rect(screen, (250, 250, 250), middle_panel); screen.set_clip(middle_panel)
        content_y = 20 - self.editor_scroll; draw_text("Agent Name:", 340, content_y)
        new_name, name_h = self.text_input("agent_name", self.selected, event, 460, content_y-5, 400); self.field_edit["agent_name"] = new_name; content_y += name_h + 10
        draw_text(f"Agent Properties ({agent_type}):", 340, content_y); content_y += 30; draw_text("help:", 340, content_y)
        val, h = self.text_input("agent_help", self.editing.get("help", ""), event, 460, content_y, 500); self.editing["help"] = val; content_y += h + 10
        if agent_type == 'template':
            draw_text("prompt:", 340, content_y); val, h = self.text_input("agent_prompt", self.editing.get("prompt", ""), event, 460, content_y, 600, 2); self.editing["prompt"] = val; content_y += h + 10
        elif agent_type == 'proc':
            draw_text("function:", 340, content_y); val, h = self.text_input("agent_function", self.editing.get("function", ""), event, 460, content_y, 600); self.editing["function"] = val; content_y += h + 10
            draw_text("function_def:", 340, content_y); y_code = content_y; code_rect = pygame.Rect(460, y_code, middle_panel_width - 140, 400)
            if self.code_editor is None: self.code_editor = CodeEditor(code_rect, self.editing.get("function_def", ""))
            self.code_editor.rect = code_rect; self.code_editor_v_thumb, self.code_editor_h_thumb = self.code_editor.draw()
            if event and event.type == pygame.MOUSEBUTTONDOWN and code_rect.collidepoint(event.pos): self.field_edit['active'] = None; self.code_editor.handle_click(event)
            self.editing["function_def"] = self.code_editor.get_value(); content_y += code_rect.height + 10
        for key in ["inputs", "optional_inputs", "outputs"]: content_y = self.draw_list_editor(f"agent_{key}", self.editing.setdefault(key, []), event, 340, content_y, middle_panel_width - 20)
        screen.set_clip(None); self.editor_panel_thumb = self.draw_scrollbar(self.editor_scroll, content_y, 340 + middle_panel_width - 10, 0, middle_panel.height)
    def draw_editor(self, event):
        if not self.editing: return
        agent_type = self.editing.get("type")
        if agent_type == "workflow": self.draw_workflow(event)
        else: self.draw_agent_fields(event)
        if self.side_panel: self.side_panel.draw(screen, event)
        save_btn, cancel_btn, delete_btn = button("Save", WIDTH - 180, HEIGHT - 60, 70), button("Cancel", WIDTH - 90, HEIGHT - 60, 70), button("Delete Agent", WIDTH - 300, HEIGHT - 60, 110, 30, RED)
        if event and event.type == pygame.MOUSEBUTTONDOWN:
            if save_btn.collidepoint(event.pos):
                self.close_side_panel()
                new_name = self.field_edit.get("agent_name", self.selected)
                if new_name != self.selected:
                    if new_name in self.agents: print(f"ERROR: Agent name '{new_name}' already exists."); return
                    else: del self.agents[self.selected]; self.selected = new_name
                self.agents[self.selected] = self.editing
                save_config(self.cfg); self.editing, self.selected, self.code_editor, self.side_panel = None, None, None, None
            elif cancel_btn.collidepoint(event.pos): self.editing, self.selected, self.code_editor, self.side_panel = None, None, None, None
            elif delete_btn.collidepoint(event.pos):
                if self.selected in self.agents: del self.agents[self.selected]; self.cfg['agents'] = self.agents; save_config(self.cfg)
                self.editing, self.selected, self.code_editor, self.side_panel = None, None, None, None
    def handle_typing(self, e):
        if (self.side_panel and self.side_panel.is_active()): return
        if (self.code_editor and not self.field_edit.get("active")): return
        active = self.field_edit.get("active");
        if not active: return
        text, cursor_pos = self.field_edit.get(active, ""), self.cursor_positions.get(active, 0)
        if e.key == pygame.K_BACKSPACE:
            if cursor_pos > 0: self.field_edit[active], self.cursor_positions[active] = text[:cursor_pos - 1] + text[cursor_pos:], cursor_pos - 1
        elif e.key == pygame.K_DELETE: self.field_edit[active] = text[:cursor_pos] + text[cursor_pos + 1:]
        elif e.key == pygame.K_LEFT: self.cursor_positions[active] = max(0, cursor_pos - 1)
        elif e.key == pygame.K_RIGHT: self.cursor_positions[active] = min(len(text), cursor_pos + 1)
        elif e.key == pygame.K_RETURN: self.field_edit[active], self.cursor_positions[active] = text[:cursor_pos] + '\n' + text[cursor_pos:], cursor_pos + 1
        elif e.key == pygame.K_ESCAPE: self.field_edit["active"] = None
        elif e.key == pygame.K_TAB: self.field_edit[active] = text[:cursor_pos] + '    ' + text[cursor_pos:]; self.cursor_positions[active] = cursor_pos + 4
        elif e.unicode: self.field_edit[active], self.cursor_positions[active] = text[:cursor_pos] + e.unicode + text[cursor_pos:], cursor_pos + len(e.unicode)
    def run(self):
        while True:
            screen.fill(WHITE); click_event = None
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(), sys.exit()
                if self.side_panel: self.side_panel.handle_event(e)
                if self.code_editor: self.code_editor.handle_event(e)
                if e.type == pygame.KEYDOWN: self.handle_typing(e)
                elif e.type == pygame.MOUSEBUTTONDOWN:
                    click_event = e
                    if e.button == 1:
                        if self.agent_list_thumb and self.agent_list_thumb.collidepoint(e.pos): self.dragging_scrollbar = 'agent'
                        elif self.editor_panel_thumb and self.editor_panel_thumb.collidepoint(e.pos): self.dragging_scrollbar = 'editor'
                        elif self.side_panel and self.side_panel.thumb_rect and self.side_panel.thumb_rect.collidepoint(e.pos): self.dragging_scrollbar = 'step'
                        elif self.code_editor and self.code_editor_v_thumb and self.code_editor_v_thumb.collidepoint(e.pos): self.dragging_scrollbar = 'code_v'
                        elif self.code_editor and self.code_editor_h_thumb and self.code_editor_h_thumb.collidepoint(e.pos): self.dragging_scrollbar = 'code_h'
                        if self.dragging_scrollbar:
                            self.drag_start_y, self.drag_start_x = e.pos[1], e.pos[0]
                            scroll_data = {'agent': self.agent_scroll, 'editor': self.editor_scroll}
                            if self.side_panel and self.dragging_scrollbar == 'step': scroll_data['step'] = self.side_panel.scroll_y
                            if self.code_editor: scroll_data.update({'code_v': self.code_editor.scroll_y, 'code_h': self.code_editor.scroll_x})
                            self.drag_start_scroll = scroll_data
                        if self.editing and self.editing.get("type") == "workflow" and 340 <= e.pos[0] < 1080 and not (self.side_panel and self.side_panel.rect.collidepoint(e.pos)):
                            self.dragging_step = self.active_idx
                    elif e.button in [4, 5]:
                        mx, my = e.pos
                        if not (self.side_panel and self.side_panel.rect.collidepoint(mx, my)) and not (self.code_editor and self.code_editor.rect.collidepoint(mx,my)):
                            scroll_amount = 30 if e.button == 5 else -30
                            if mx < 320: self.agent_scroll = max(0, self.agent_scroll + scroll_amount)
                            elif 340 <= mx < 1080 and self.editing: self.editor_scroll = max(0, self.editor_scroll + scroll_amount)
                elif e.type == pygame.MOUSEMOTION:
                    if self.dragging_scrollbar:
                        dy = e.pos[1] - self.drag_start_y
                        if self.dragging_scrollbar == 'step' and self.side_panel and self.side_panel.thumb_rect:
                            total_h, area_h = self.side_panel.content_height, self.side_panel.rect.height
                            if total_h > area_h:
                                scroll_change = dy * (total_h - area_h) / (area_h - self.side_panel.thumb_rect.height)
                                self.side_panel.scroll_y = max(0, min(self.drag_start_scroll['step'] + scroll_change, total_h - area_h))
                        # Other scrollbar logic...
                elif e.type == pygame.MOUSEBUTTONUP:
                    self.dragging_scrollbar = None
                    if self.dragging_step is not None:
                        if self.drop_target is not None and self.drop_target != self.dragging_step and self.drop_target < len(self.editing["steps"]):
                            self.editing["steps"].insert(self.drop_target, self.editing["steps"].pop(self.dragging_step))
                            if self.active_idx == self.dragging_step:
                                self.close_side_panel()
                                self.open_side_panel(self.drop_target)
                        self.dragging_step, self.drop_target = None, None
            self.draw_agent_list(click_event)
            self.draw_editor(click_event)
            pygame.display.flip()
            clock.tick(30)

if __name__ == "__main__":
    Editor().run()
