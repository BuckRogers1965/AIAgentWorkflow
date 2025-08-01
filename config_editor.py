import pygame, json, sys, os
from pygments import lex
from pygments.lexers import PythonLexer
from pygments.token import Token
from pygments.styles import get_style_by_name

# --- Constants and Initialization ---
CONFIG_PATH = "config.json"
WIDTH, HEIGHT = 1400, 800
WHITE, BLACK, GRAY, BLUE, GREEN, RED, DARK_GRAY, LIGHT_BLUE = (255,255,255), (0,0,0), (180,180,180), (0,120,255), (0,180,0), (180,0,0), (100,100,100), (200,220,255)
MODAL_BG_COLOR = (245, 245, 245)
TOOLBOX_BG_COLOR = (235, 235, 240)

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Agent Workflow Editor")
font = pygame.font.SysFont(None, 22)
small_font = pygame.font.SysFont(None, 18)
code_font = pygame.font.SysFont("monospace", 16)
clock = pygame.time.Clock()

# --- Pygments Setup ---
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

# --- Utility Functions ---
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

def button(txt, x, y, w=250, h=30, color=GRAY, surface=None):
    if surface is None: surface = screen
    r = pygame.Rect(x, y, w, h); pygame.draw.rect(surface, color, r); draw_text(txt, x + 6, y + 6, surface=surface)
    return r

def small_button(txt, x, y, w=60, h=20, color=GRAY, surface=None):
    if surface is None: surface = screen
    r = pygame.Rect(x, y, w, h); pygame.draw.rect(surface, color, r); draw_text(txt, x + 4, y + 2, BLACK, small_font, surface=surface)
    return r

def wrap_text(text, width, font_obj):
    lines, text = [], str(text)
    for paragraph in text.split('\n'):
        words, current_line = paragraph.split(' '), ""
        for word in words:
            if font_obj.size(word)[0] > width:
                if current_line: lines.append(current_line)
                current_line, temp_word = "", ""
                for char in word:
                    if font_obj.size(temp_word + char)[0] <= width: temp_word += char
                    else: lines.append(temp_word); temp_word = char
                current_line = temp_word
            else:
                test_line = current_line + (' ' if current_line else '') + word
                if font_obj.size(test_line)[0] <= width: current_line = test_line
                else: lines.append(current_line); current_line = word
        lines.append(current_line)
    return lines if lines else [""]

def draw_scrollbar(surface, scroll_y, content_height, area_rect):
    if content_height <= area_rect.height: return None
    bar_rect = pygame.Rect(area_rect.right - 12, area_rect.y, 10, area_rect.height)
    thumb_h = max(20, area_rect.height * (area_rect.height / content_height))
    thumb_y = area_rect.y + (scroll_y / (content_height - area_rect.height)) * (area_rect.height - thumb_h) if content_height > area_rect.height else area_rect.y
    thumb_rect = pygame.Rect(bar_rect.x, thumb_y, 10, thumb_h)
    pygame.draw.rect(surface, GRAY, bar_rect)
    pygame.draw.rect(surface, DARK_GRAY, thumb_rect)
    return thumb_rect

# --- StepEditorModal Class (REFACTORED & FIXED) ---
class StepEditorModal:
    def __init__(self, step_index, step_data, all_agents_data):
        self.index = step_index
        self.editing_data = json.loads(json.dumps(step_data))
        self.agent_name = self.editing_data.get("agent")
        self.agent_master_def = all_agents_data.get(self.agent_name, {})
        
        self.rect = pygame.Rect(0, 0, 900, 600); self.rect.center = (WIDTH // 2, HEIGHT // 2)
        self.toolbox_rect = pygame.Rect(self.rect.x, self.rect.y, 250, self.rect.height)
        self.form_rect = pygame.Rect(self.rect.x + 250, self.rect.y, self.rect.width - 250, self.rect.height)
        
        self.toolbox_scroll_y, self.form_scroll_y = 0, 0
        self.toolbox_content_h, self.form_content_h = 0, 0
        self.active_field, self.cursor_positions = None, {}
        
        self.last_click_event = None
        self.save_button_rect, self.cancel_button_rect = None, None
        self.toolbox_thumb_rect, self.form_thumb_rect = None, None
        self.dragging_scrollbar = None
        self.ui_rects = {}

    def _add_optional_param(self):
        if 'params' not in self.editing_data: self.editing_data['params'] = {}
        i = 0
        while f"param_{i}" in self.editing_data['params']: i += 1
        self.editing_data['params'][f"param_{i}"] = ""

    def _remove_optional_param(self, param_name):
        if 'params' in self.editing_data and param_name in self.editing_data['params']:
            del self.editing_data['params'][param_name]

    def _text_input(self, surface, event, field_id, value, x, y, w, is_editable=True):
        text = str(value); lines = wrap_text(text, w - 12, font); line_height = 22
        h = max(30, len(lines) * line_height + 8); r = pygame.Rect(x, y, w, h)
        is_active = (self.active_field == field_id)
        
        pygame.draw.rect(surface, (230,230,230) if not is_editable else WHITE, r)
        pygame.draw.rect(surface, LIGHT_BLUE if is_active else BLACK, r, 2 if is_active else 1)
        
        for i, line in enumerate(lines): draw_text(line, x + 6, y + 6 + i * line_height, BLACK, font, surface)
        if is_active and pygame.time.get_ticks() % 1000 < 500:
            cursor_pos = self.cursor_positions.get(field_id, len(text)); text_upto_cursor = text[:cursor_pos]
            wrapped_upto_cursor = wrap_text(text_upto_cursor, w-12, font)
            cursor_line_idx = max(0, len(wrapped_upto_cursor) - 1); cursor_x_offset = font.size(wrapped_upto_cursor[-1])[0]
            cursor_x = x + 6 + cursor_x_offset; cursor_y = y + 6 + cursor_line_idx * line_height
            pygame.draw.line(surface, BLACK, (cursor_x, cursor_y), (cursor_x, cursor_y + 18), 1)
        
        if event and event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and r.collidepoint(event.pos):
            if is_editable: self.active_field = field_id
            mx, my = event.pos[0], event.pos[1]; clicked_line_idx = max(0, min(len(lines) - 1, (my - y - 6) // line_height))
            click_x_in_line = mx - (x + 6); text_lines = text.split('\n'); abs_pos, temp_wrapped_lines, found_line = 0, [], False
            for p_idx, p in enumerate(text_lines):
                wrapped_p = wrap_text(p, w - 12, font)
                if len(temp_wrapped_lines) + len(wrapped_p) > clicked_line_idx:
                    sub_line_idx = clicked_line_idx - len(temp_wrapped_lines); line_text = wrapped_p[sub_line_idx]
                    min_dist, best_col = float('inf'), 0
                    for i in range(len(line_text) + 1):
                        dist = abs(click_x_in_line - font.size(line_text[:i])[0])
                        if dist < min_dist: min_dist, best_col = dist, i
                    abs_pos += sum(len(l) for l in wrapped_p[:sub_line_idx]); abs_pos += sub_line_idx; abs_pos += best_col; found_line = True; break
                else: temp_wrapped_lines.extend(wrapped_p); abs_pos += len(p)
                if p_idx < len(text_lines) - 1: abs_pos += 1
            if not found_line: abs_pos = len(text)
            self.cursor_positions[field_id] = min(abs_pos, len(text))
        return h

    def _handle_typing(self, e):
        if not self.active_field: return
        field_type, field_key = self.active_field; text = ""
        
        if field_type == 'output': text = self.editing_data.get('output', [])[field_key]
        elif field_type == 'param_value': text = self.editing_data.get('params', {}).get(field_key, '')
        elif field_type == 'param_key': text = field_key
        
        cursor_pos = self.cursor_positions.get(self.active_field, 0); new_text = str(text)
        
        if e.key == pygame.K_BACKSPACE:
            if cursor_pos > 0: new_text = new_text[:cursor_pos-1] + new_text[cursor_pos:]; cursor_pos -= 1
        elif e.key == pygame.K_DELETE: new_text = new_text[:cursor_pos] + new_text[cursor_pos+1:]
        elif e.key == pygame.K_LEFT: cursor_pos = max(0, cursor_pos-1)
        elif e.key == pygame.K_RIGHT: cursor_pos = min(len(new_text), cursor_pos+1)
        elif e.key == pygame.K_ESCAPE: self.active_field = None; return
        elif e.key in [pygame.K_RETURN, pygame.K_KP_ENTER]: new_text = new_text[:cursor_pos] + '\n' + new_text[cursor_pos:]; cursor_pos += 1
        elif e.unicode: new_text = new_text[:cursor_pos] + e.unicode + new_text[cursor_pos:]; cursor_pos += len(e.unicode)
        
        self.cursor_positions[self.active_field] = cursor_pos
        
        if field_type == 'output': self.editing_data['output'][field_key] = new_text
        elif field_type == 'param_value': self.editing_data['params'][field_key] = new_text
        elif field_type == 'param_key':
            if new_text != field_key and new_text and new_text not in self.editing_data['params']:
                self.editing_data['params'][new_text] = self.editing_data['params'].pop(field_key)
                new_active_field = ('param_key', new_text)
                self.cursor_positions[new_active_field] = self.cursor_positions.pop(self.active_field, cursor_pos)
                self.active_field = new_active_field

    def draw(self, main_surface):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 128)); main_surface.blit(overlay, (0, 0))
        pygame.draw.rect(main_surface, MODAL_BG_COLOR, self.rect); pygame.draw.rect(main_surface, BLACK, self.rect, 2)
        self.ui_rects = {}
        self.draw_toolbox(main_surface); self.draw_form(main_surface); self.draw_action_bar(main_surface)
        self.toolbox_thumb_rect = draw_scrollbar(main_surface, self.toolbox_scroll_y, self.toolbox_content_h, self.toolbox_rect)
        self.form_thumb_rect = draw_scrollbar(main_surface, self.form_scroll_y, self.form_content_h, self.form_rect)

    def draw_toolbox(self, surface):
        toolbox_surf = pygame.Surface(self.toolbox_rect.size); toolbox_surf.fill(TOOLBOX_BG_COLOR); y, x, w = 10, 10, self.toolbox_rect.width - 20
        draw_text("Optional Inputs", x, y, surface=toolbox_surf); y += 30
        for param in self.agent_master_def.get("optional_inputs", []):
            color = (100, 200, 100); add_btn_rect = small_button(f"+ {param}", x, y, w, 25, color, toolbox_surf)
            # FIX: Store the absolute on-screen rect, accounting for scroll
            stored_rect = add_btn_rect.move(self.toolbox_rect.x, self.toolbox_rect.y - self.toolbox_scroll_y)
            self.ui_rects[f'add_opt_{param}'] = stored_rect
            y += 30
        self.toolbox_content_h = y
        surface.blit(toolbox_surf, self.toolbox_rect.topleft, (0, self.toolbox_scroll_y, self.toolbox_rect.width, self.toolbox_rect.height))

    def _draw_dict_editor(self, surface, event, title, data_dict, editable_keys_list, x, y, w):
        draw_text(title, x, y, surface=surface); y += 25; key_w, val_w = 150, w - 180
        for key, value in list(data_dict.items()):
            is_key_editable = key in editable_keys_list
            key_h = self._text_input(surface, event, ('param_key', key), key, x + 20, y, key_w, is_key_editable)
            val_h = self._text_input(surface, event, ('param_value', key), value, x + 30 + key_w, y, val_w)
            row_h = max(key_h, val_h)
            if is_key_editable:
                remove_btn = small_button("X", x + w - 25, y + (row_h - 20)//2, 20, 20, RED, surface)
                # FIX: Store the absolute on-screen rect, accounting for scroll
                stored_rect = remove_btn.move(self.form_rect.x, self.form_rect.y - self.form_scroll_y)
                self.ui_rects[f'remove_opt_{key}'] = stored_rect
            y += row_h + 5
        return y

    def draw_form(self, surface):
        form_surf = pygame.Surface((self.form_rect.width, 3000)); form_surf.fill(MODAL_BG_COLOR)
        event_for_ui = self.last_click_event if self.last_click_event else None
        y, x, w = 10, 10, self.form_rect.width - 40
        draw_text(f"Editing Step {self.index}: {self.agent_name}", x, y, surface=form_surf); y += 30
        draw_text("Output Mapping:", x, y, surface=form_surf); y += 25
        outputs = self.editing_data.setdefault('output', [])
        for i, item in enumerate(outputs):
            h = self._text_input(form_surf, event_for_ui, ('output', i), item, x + 20, y, w - 20); y += h + 5
        y += 15
        
        all_params = self.editing_data.setdefault('params', {})
        required_params = {k:v for k,v in all_params.items() if k in self.agent_master_def.get("inputs", [])}
        optional_params = {k:v for k,v in all_params.items() if k not in self.agent_master_def.get("inputs", [])}
        
        y = self._draw_dict_editor(form_surf, event_for_ui, "Required Parameters:", required_params, [], x, y, w)
        y += 15
        y = self._draw_dict_editor(form_surf, event_for_ui, "Optional Parameters:", optional_params, list(optional_params.keys()), x, y, w)
        add_param_btn = small_button("+ Add Parameter", x + 20, y, 120, 25, GREEN, form_surf)
        stored_rect = add_param_btn.move(self.form_rect.x, self.form_rect.y - self.form_scroll_y)
        self.ui_rects['add_new_param'] = stored_rect
        y += 35

        self.form_content_h = y
        surface.blit(form_surf, self.form_rect.topleft, (0, self.form_scroll_y, self.form_rect.width, self.form_rect.height))

    def draw_action_bar(self, surface):
        bar_y = self.rect.bottom - 50
        self.cancel_button_rect = button("Cancel", self.rect.x + 20, bar_y, 100, 40, surface=surface)
        self.save_button_rect = button("Save", self.rect.right - 120, bar_y, 100, 40, GREEN, surface=surface)

    def handle_event(self, event):
        self.last_click_event = None
        if event.type == pygame.KEYDOWN: self._handle_typing(event)
        elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if not self.rect.collidepoint(event.pos): return {'status': 'cancel'}
            self.last_click_event = pygame.event.Event(event.type, {'pos': (event.pos[0] - self.form_rect.x, event.pos[1] - self.form_rect.y + self.form_scroll_y), 'button': 1})
            if self.save_button_rect and self.save_button_rect.collidepoint(event.pos): return {'status': 'save', 'data': self.editing_data}
            if self.cancel_button_rect and self.cancel_button_rect.collidepoint(event.pos): return {'status': 'cancel'}
            if self.toolbox_thumb_rect and self.toolbox_thumb_rect.collidepoint(event.pos): self.dragging_scrollbar = 'toolbox'; self.drag_start_y = event.pos[1]; self.drag_start_scroll = self.toolbox_scroll_y
            elif self.form_thumb_rect and self.form_thumb_rect.collidepoint(event.pos): self.dragging_scrollbar = 'form'; self.drag_start_y = event.pos[1]; self.drag_start_scroll = self.form_scroll_y
            else:
                for key, rect in self.ui_rects.items():
                    if rect.collidepoint(event.pos):
                        if key.startswith('add_opt_'): self.editing_data.setdefault('params', {})[key[8:]]=""; break
                        elif key == 'add_new_param': self._add_optional_param(); break
                        elif key.startswith('remove_opt_'): self._remove_optional_param(key[11:]); break
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1: self.dragging_scrollbar = None
        elif event.type == pygame.MOUSEMOTION:
            if self.dragging_scrollbar:
                dy = event.pos[1] - self.drag_start_y
                if self.dragging_scrollbar == 'toolbox' and self.toolbox_thumb_rect:
                    max_scroll = self.toolbox_content_h - self.toolbox_rect.height
                    if max_scroll > 0:
                        scroll_ratio = (self.toolbox_rect.height - self.toolbox_thumb_rect.height) / max_scroll
                        self.toolbox_scroll_y = max(0, min(max_scroll, self.drag_start_scroll + dy / scroll_ratio if scroll_ratio > 0 else 0))
                elif self.dragging_scrollbar == 'form' and self.form_thumb_rect:
                    max_scroll = self.form_content_h - self.form_rect.height
                    if max_scroll > 0:
                        scroll_ratio = (self.form_rect.height - self.form_thumb_rect.height) / max_scroll
                        self.form_scroll_y = max(0, min(max_scroll, self.drag_start_scroll + dy / scroll_ratio if scroll_ratio > 0 else 0))
        elif event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos()
            if self.rect.collidepoint(mouse_pos):
                scroll_amount = -event.y * 20
                if self.toolbox_rect.collidepoint(mouse_pos):
                    max_scroll = self.toolbox_content_h - self.toolbox_rect.height
                    self.toolbox_scroll_y = max(0, min(self.toolbox_scroll_y + scroll_amount, max_scroll if max_scroll > 0 else 0))
                elif self.form_rect.collidepoint(mouse_pos):
                    max_scroll = self.form_content_h - self.form_rect.height
                    self.form_scroll_y = max(0, min(self.form_scroll_y + scroll_amount, max_scroll if max_scroll > 0 else 0))
        return None

# --- Main Editor Class ---
class Editor:
    ALLOWED_AGENT_TYPES = {"workflow", "template", "proc"}
    def __init__(self):
        self.cfg = load_config(); self.agents = self.cfg.get("agents", {})
        self.selected, self.editing = None, None
        self.agent_scroll, self.editor_scroll = 0, 0
        self.dragging_step, self.drop_target = None, None
        self.field_edit, self.cursor_positions, self.code_editor = {}, {}, None
        self.dragging_scrollbar = None
        self.agent_list_thumb, self.editor_panel_thumb = None, None
        self.step_modal, self.pending_deletion_agent = None, None

    def open_step_editor(self, index):
        if self.editing and 0 <= index < len(self.editing.get("steps", [])): self.step_modal = StepEditorModal(index, self.editing["steps"][index], self.agents)
    
    def close_step_editor(self, result=None):
        if result and result.get('status') == 'save': self.editing["steps"][self.step_modal.index] = result['data']
        self.step_modal = None
        
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
        self.agents[name] = new_agent; self.selected, self.editing = name, json.loads(json.dumps(self.agents[name]))
        self.field_edit, self.code_editor, self.step_modal = {}, None, None
    
    def add_agent_as_step(self, agent_name):
        if self.editing and self.editing.get("type") == "workflow":
            agent_info = self.agents.get(agent_name, {})
            new_params = {key: f"${key}" for key in agent_info.get("inputs", [])}
            new_step = {"agent": agent_name, "params": new_params, "output": agent_info.get("outputs", ["output"]).copy()}
            self.editing["steps"].append(new_step)
    
    def remove_workflow_step(self, idx):
        if self.editing and self.editing.get("type") == "workflow" and 0 <= idx < len(self.editing["steps"]):
            if self.step_modal and self.step_modal.index == idx: self.close_step_editor()
            self.editing["steps"].pop(idx)

    def text_input(self, key, val, event, x, y, w=300, min_lines=1):
        if self.step_modal: self.field_edit['active'] = None
        if key not in self.field_edit: self.field_edit[key] = str(val)
        text = self.field_edit[key]; lines = wrap_text(text, w - 12, font)
        while len(lines) < min_lines: lines.append("")
        line_height = 22; h = max(30, len(lines) * line_height + 8); r = pygame.Rect(x, y, w, h)
        pygame.draw.rect(screen, WHITE, r); pygame.draw.rect(screen, BLACK, r, 1); is_active = self.field_edit.get("active") == key
        if is_active: pygame.draw.rect(screen, LIGHT_BLUE, r, 2)
        for i, line in enumerate(lines): draw_text(line, x + 6, y + 6 + i * line_height)
        if is_active and pygame.time.get_ticks() % 1000 < 500:
            cursor_pos = self.cursor_positions.get(key, len(text)); text_upto_cursor = text[:cursor_pos]; wrapped_upto_cursor = wrap_text(text_upto_cursor, w-12, font)
            cursor_line_idx = max(0, len(wrapped_upto_cursor) - 1); cursor_x_offset = font.size(wrapped_upto_cursor[-1])[0]
            cursor_x = x + 6 + cursor_x_offset; cursor_y = y + 6 + cursor_line_idx * line_height
            pygame.draw.line(screen, BLACK, (cursor_x, cursor_y), (cursor_x, cursor_y + 18), 1)
        if event and event.type == pygame.MOUSEBUTTONDOWN and r.collidepoint(event.pos):
            self.field_edit["active"] = key; self.code_editor = None
            if self.step_modal: self.step_modal.active_field = None
            mx, my = event.pos; clicked_line_idx = max(0, min(len(lines) - 1, (my - y - 6) // line_height))
            click_x_in_line = mx - (x + 6); text_lines = text.split('\n'); abs_pos, temp_wrapped_lines, found_line = 0, [], False
            for p_idx, p in enumerate(text_lines):
                wrapped_p = wrap_text(p, w - 12, font)
                if len(temp_wrapped_lines) + len(wrapped_p) > clicked_line_idx:
                    sub_line_idx = clicked_line_idx - len(temp_wrapped_lines); line_text = wrapped_p[sub_line_idx]; min_dist, best_col = float('inf'), 0
                    for i in range(len(line_text) + 1):
                        dist = abs(click_x_in_line - font.size(line_text[:i])[0])
                        if dist < min_dist: min_dist, best_col = dist, i
                    abs_pos += sum(len(l.replace(' ', '')) for l in wrapped_p[:sub_line_idx]); abs_pos += best_col; found_line = True; break
                else: temp_wrapped_lines.extend(wrapped_p); abs_pos += len(p)
                if p_idx < len(text_lines) - 1: abs_pos += 1
            if not found_line: abs_pos = len(text)
            self.cursor_positions[key] = min(abs_pos, len(text))
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

    def draw_checkbox(self, value, event, x, y):
        checkbox_rect = pygame.Rect(x, y, 16, 16); pygame.draw.rect(screen, WHITE, checkbox_rect); pygame.draw.rect(screen, BLACK, checkbox_rect, 1)
        if value: pygame.draw.lines(screen, BLACK, False, [(x+3,y+8),(x+7,y+12),(x+13,y+4)], 2)
        if event and event.type == pygame.MOUSEBUTTONDOWN and checkbox_rect.collidepoint(event.pos): return not value
        return value

    def draw_agent_list(self, event):
        start_y, button_h, spacing = 20, 30, 10; draw_text("Agents:", 20, 0)
        add_buttons = [("Add Workflow", "workflow"), ("Add Template", "template"), ("Add Proc", "proc")]
        for i, (label, agent_type) in enumerate(add_buttons):
            r = button(label, 20, start_y + i * (button_h + spacing), 250, 30, surface=screen)
            if event and event.type == pygame.MOUSEBUTTONDOWN and r.collidepoint(event.pos): self.add_agent(agent_type)
        agent_y_start = start_y + len(add_buttons) * (button_h + spacing) + spacing; view_height = HEIGHT - agent_y_start - 10; y = agent_y_start - self.agent_scroll
        if self.editing and self.editing.get("type") == "workflow": draw_text("Click agent to add as step:", 20, agent_y_start - 20, BLUE, small_font)
        clip_rect = pygame.Rect(20, agent_y_start, 260, view_height); screen.set_clip(clip_rect)
        sorted_keys = sorted(self.agents.keys()); content_height = 0
        for k in sorted_keys:
            agent_type = self.agents.get(k, {}).get("type"); is_valid_type = agent_type in self.ALLOWED_AGENT_TYPES
            color = BLUE if k == self.selected else (GRAY if is_valid_type else DARK_GRAY)
            r = button(f"[{agent_type[0] if agent_type else '?'}] {k}", 20, y, 220, 30, color, surface=screen)
            delete_btn = small_button("X", 245, y + 5, 20, 20, RED, surface=screen)
            if event and event.type == pygame.MOUSEBUTTONDOWN:
                if r.collidepoint(event.pos) and is_valid_type:
                    if self.editing and self.editing.get("type") == "workflow": self.add_agent_as_step(k)
                    else: self.step_modal = None; self.selected, self.editing = k, json.loads(json.dumps(self.agents[k])); self.field_edit, self.code_editor = {}, None
                elif delete_btn.collidepoint(event.pos) and k != self.selected: self.pending_deletion_agent = k
            y += 35; content_height += 35
        screen.set_clip(None); self.agent_list_thumb = draw_scrollbar(screen, self.agent_scroll, content_height, pygame.Rect(290, agent_y_start, 10, view_height))

    def draw_workflow(self, event):
        middle_panel_width = 620; middle_panel = pygame.Rect(340, 0, middle_panel_width, HEIGHT)
        pygame.draw.rect(screen, (250, 250, 250), middle_panel); screen.set_clip(middle_panel)
        content_y = 20 - self.editor_scroll
        draw_text("Agent Name:", 340, content_y); new_name, name_h = self.text_input("agent_name", self.selected, event, 460, content_y-5, 400); self.field_edit["agent_name"] = new_name; content_y += name_h + 10
        draw_text("Workflow Properties:", 340, content_y); content_y += 30
        draw_text("help:", 340, content_y); val, h = self.text_input("wf_help", self.editing.get("help", ""), event, 460, content_y, 500); self.editing["help"] = val; content_y += h + 10
        draw_text("return_on_fail:", 340, content_y); self.editing["return_on_fail"] = self.draw_checkbox(self.editing.get("return_on_fail", 0), event, 460, content_y); content_y += 30
        for key in ["inputs", "optional_inputs", "outputs"]: content_y = self.draw_list_editor(f"wf_{key}", self.editing.setdefault(key, []), event, 340, content_y, middle_panel_width - 20)
        content_y += 20; draw_text("Workflow Steps:", 340, content_y); content_y += 40
        step_to_remove = -1; steps = self.editing.get("steps", []); self.drop_target = None
        for i, step in enumerate(steps):
            step_row_y = content_y
            if i == self.dragging_step: content_y += 40; continue
            step_rect = pygame.Rect(340, step_row_y, middle_panel_width - 20, 30); pygame.draw.rect(screen, GRAY, step_rect)
            edit_btn = small_button("Edit", 345, step_row_y + 5, 40, 20, color=GREEN)
            draw_text(f"{i}. {step.get('agent', 'unknown')}", 395, step_row_y + 6)
            remove_btn = small_button("X", step_rect.right - 25, step_row_y + 5, 20, 20, RED)
            if event and event.type == pygame.MOUSEBUTTONDOWN:
                if edit_btn.collidepoint(event.pos): self.open_step_editor(i)
                elif remove_btn.collidepoint(event.pos): step_to_remove = i
                elif step_rect.collidepoint(event.pos): self.dragging_step = i
            mouse_pos = pygame.mouse.get_pos()
            if self.dragging_step is not None and step_rect.collidepoint(mouse_pos): self.drop_target = i if mouse_pos[1] < step_rect.centery else i + 1
            if self.dragging_step is not None and self.drop_target == i: pygame.draw.line(screen, RED, (step_rect.left + 5, step_row_y - 2), (step_rect.right - 5, step_row_y - 2), 3)
            content_y += 40
        if self.dragging_step is not None:
            if self.drop_target is None and pygame.mouse.get_pos()[1] > content_y - self.editor_scroll: self.drop_target = len(steps)
            if self.drop_target == len(steps): pygame.draw.line(screen, RED, (345, content_y - 2), (middle_panel.right - 25, content_y - 2), 3)
        if step_to_remove != -1: self.remove_workflow_step(step_to_remove)
        screen.set_clip(None)
        if self.dragging_step is not None and self.dragging_step < len(steps):
            dragged_step_data = self.editing["steps"][self.dragging_step]; mouse_x, mouse_y = pygame.mouse.get_pos()
            drag_surf_width = middle_panel_width - 20; drag_surf = pygame.Surface((drag_surf_width, 30), pygame.SRCALPHA); drag_surf.set_alpha(180)
            pygame.draw.rect(drag_surf, BLUE, (0, 0, drag_surf_width, 30)); draw_text(f"{self.dragging_step}. {dragged_step_data.get('agent', 'unknown')}", 55, 6, surface=drag_surf)
            screen.blit(drag_surf, (mouse_x - drag_surf_width // 2, mouse_y - 15))
        self.editor_panel_thumb = draw_scrollbar(screen, self.editor_scroll, content_y, middle_panel)

    def draw_agent_fields(self, event):
        agent_type = self.editing['type']; middle_panel_width = 1040 if agent_type == 'proc' else 600
        middle_panel = pygame.Rect(340, 0, middle_panel_width, HEIGHT); pygame.draw.rect(screen, (250, 250, 250), middle_panel)
        screen.set_clip(middle_panel); content_y = 20 - self.editor_scroll
        draw_text("Agent Name:", 340, content_y); new_name, name_h = self.text_input("agent_name", self.selected, event, 460, content_y-5, 400); self.field_edit["agent_name"] = new_name; content_y += name_h + 10
        draw_text(f"Agent Properties ({agent_type}):", 340, content_y); content_y += 30
        draw_text("help:", 340, content_y); val, h = self.text_input("agent_help", self.editing.get("help", ""), event, 460, content_y, 500); self.editing["help"] = val; content_y += h + 10
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
        screen.set_clip(None); self.editor_panel_thumb = draw_scrollbar(screen, self.editor_scroll, content_y, middle_panel)

    def draw_editor(self, event):
        if not self.editing: return
        agent_type = self.editing.get("type")
        if agent_type == "workflow": self.draw_workflow(event)
        else: self.draw_agent_fields(event)
        save_btn = button("Save", WIDTH - 180, HEIGHT - 60, 70, surface=screen)
        cancel_btn = button("Cancel", WIDTH - 90, HEIGHT - 60, 70, surface=screen)
        delete_btn = button("Delete Agent", WIDTH - 300, HEIGHT - 60, 110, 30, RED, surface=screen)
        if event and event.type == pygame.MOUSEBUTTONDOWN:
            if save_btn.collidepoint(event.pos):
                self.step_modal = None; new_name = self.field_edit.get("agent_name", self.selected)
                if new_name != self.selected:
                    if new_name in self.agents: print(f"ERROR: Agent name '{new_name}' already exists."); return
                    else: del self.agents[self.selected]; self.selected = new_name
                self.agents[self.selected] = self.editing; save_config(self.cfg); self.editing, self.selected, self.code_editor = None, None, None
            elif cancel_btn.collidepoint(event.pos): self.editing, self.selected, self.code_editor, self.step_modal = None, None, None, None
            elif delete_btn.collidepoint(event.pos):
                if self.selected in self.agents: self.pending_deletion_agent = self.selected

    def handle_typing(self, e):
        if self.step_modal or (self.code_editor and not self.field_edit.get("active")): return
        active = self.field_edit.get("active")
        if not active: return
        text, cursor_pos = self.field_edit.get(active, ""), self.cursor_positions.get(active, 0)
        if e.key == pygame.K_BACKSPACE:
            if cursor_pos > 0: self.field_edit[active], self.cursor_positions[active] = text[:cursor_pos - 1] + text[cursor_pos:], cursor_pos - 1
        elif e.key == pygame.K_DELETE: self.field_edit[active] = text[:cursor_pos] + text[cursor_pos + 1:]
        elif e.key == pygame.K_LEFT: self.cursor_positions[active] = max(0, cursor_pos - 1)
        elif e.key == pygame.K_RIGHT: self.cursor_positions[active] = min(len(text), cursor_pos + 1)
        elif e.key in [pygame.K_RETURN, pygame.K_KP_ENTER]: self.field_edit[active], self.cursor_positions[active] = text[:cursor_pos] + '\n' + text[cursor_pos:], cursor_pos + 1
        elif e.key == pygame.K_ESCAPE: self.field_edit["active"] = None
        elif e.key == pygame.K_TAB: self.field_edit[active] = text[:cursor_pos] + '    ' + text[cursor_pos:]; self.cursor_positions[active] = cursor_pos + 4
        elif e.unicode: self.field_edit[active], self.cursor_positions[active] = text[:cursor_pos] + e.unicode + text[cursor_pos:], cursor_pos + len(e.unicode)

    def check_agent_usage(self, agent_to_check):
        dependents = []
        for agent_name, agent_data in self.agents.items():
            if agent_data.get("type") == "workflow":
                for step in agent_data.get("steps", []):
                    if step.get("agent") == agent_to_check: dependents.append(agent_name); break
        return dependents

    def draw_delete_confirmation(self):
        overlay = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA); overlay.fill((0, 0, 0, 128)); screen.blit(overlay, (0, 0))
        dependents = self.check_agent_usage(self.pending_deletion_agent)
        dialog_width = 500; base_height = 120; text_height = 25 * len(dependents) if dependents else 0
        dialog_height = base_height + text_height; dialog_x, dialog_y = (WIDTH - dialog_width) // 2, (HEIGHT - dialog_height) // 2
        pygame.draw.rect(screen, (245, 245, 245), (dialog_x, dialog_y, dialog_width, dialog_height)); pygame.draw.rect(screen, BLACK, (dialog_x, dialog_y, dialog_width, dialog_height), 2); y = dialog_y + 20
        if dependents:
            draw_text(f"Cannot delete agent '{self.pending_deletion_agent}'", dialog_x + 20, y, RED); y += 30
            draw_text("It is used by the following workflows:", dialog_x + 20, y); y += 25
            for dep in dependents: draw_text(f"  • {dep}", dialog_x + 30, y, DARK_GRAY); y += 25
            button("OK", dialog_x + dialog_width // 2 - 50, dialog_y + dialog_height - 50, 100, 30, GREEN, surface=screen)
        else:
            draw_text(f"Delete Agent '{self.pending_deletion_agent}'?", dialog_x + 20, y, RED); y += 30
            draw_text("This action cannot be undone.", dialog_x + 20, y); y += 25
            button("Cancel", dialog_x + 40, dialog_y + dialog_height - 50, 100, 30, GREEN, surface=screen)
            button("Delete", dialog_x + dialog_width - 140, dialog_y + dialog_height - 50, 100, 30, RED, surface=screen)

    def handle_delete_confirmation_events(self, event):
        if not (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1): return
        dependents = self.check_agent_usage(self.pending_deletion_agent)
        dialog_width=500; base_height=120; text_height=25*len(dependents) if dependents else 0; dialog_height=base_height+text_height; dialog_x=(WIDTH-dialog_width)//2; dialog_y=(HEIGHT-dialog_height)//2; dialog_rect = pygame.Rect(dialog_x, dialog_y, dialog_width, dialog_height)
        if dependents:
            if pygame.Rect(dialog_x + dialog_width // 2 - 50, dialog_y + dialog_height - 50, 100, 30).collidepoint(event.pos): self.pending_deletion_agent = None
        else:
            if pygame.Rect(dialog_x + 40, dialog_y + dialog_height - 50, 100, 30).collidepoint(event.pos): self.pending_deletion_agent = None
            elif pygame.Rect(dialog_x+dialog_width-140, dialog_y+dialog_height-50, 100, 30).collidepoint(event.pos):
                if self.selected == self.pending_deletion_agent: self.editing, self.selected = None, None
                del self.agents[self.pending_deletion_agent]; save_config(self.cfg); self.pending_deletion_agent = None
        if not dialog_rect.collidepoint(event.pos): self.pending_deletion_agent = None

    def run(self):
        while True:
            screen.fill(WHITE)
            click_event = None
            
            for e in pygame.event.get():
                if e.type == pygame.QUIT: pygame.quit(); sys.exit()
                
                if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                    click_event = e

                if self.step_modal:
                    result = self.step_modal.handle_event(e)
                    if result: self.close_step_editor(result)
                elif self.pending_deletion_agent:
                    self.handle_delete_confirmation_events(e)
                else:
                    if self.code_editor: self.code_editor.handle_event(e)
                    if e.type == pygame.KEYDOWN: self.handle_typing(e)
                    
                    if e.type == pygame.MOUSEBUTTONDOWN and e.button == 1:
                        if self.editing:
                            agent_list_rect = pygame.Rect(0, 0, 320, HEIGHT); agent_type = self.editing.get("type"); mpw = 1040 if agent_type == 'proc' else 620
                            editor_panel_rect = pygame.Rect(340, 0, mpw, HEIGHT); buttons_rect = pygame.Rect(WIDTH - 310, HEIGHT - 70, 300, 60)
                            if not agent_list_rect.collidepoint(e.pos) and not editor_panel_rect.collidepoint(e.pos) and not buttons_rect.collidepoint(e.pos):
                                self.editing, self.selected, self.code_editor, self.step_modal = None, None, None, None; continue
                        if self.agent_list_thumb and self.agent_list_thumb.collidepoint(e.pos): self.dragging_scrollbar = 'agent'
                        elif self.editor_panel_thumb and self.editor_panel_thumb.collidepoint(e.pos): self.dragging_scrollbar = 'editor'
                        if self.dragging_scrollbar: self.drag_start_y = e.pos[1]; self.drag_start_scroll = {'agent': self.agent_scroll, 'editor': self.editor_scroll}.get(self.dragging_scrollbar, 0)
                    
                    elif e.type == pygame.MOUSEBUTTONUP and e.button == 1:
                        self.dragging_scrollbar = None
                        if self.dragging_step is not None and self.drop_target is not None:
                            if self.drop_target != self.dragging_step and self.drop_target != self.dragging_step + 1:
                                item_to_move = self.editing["steps"].pop(self.dragging_step); new_drop_index = self.drop_target
                                if self.drop_target > self.dragging_step: new_drop_index -= 1
                                self.editing["steps"].insert(new_drop_index, item_to_move)
                        self.dragging_step, self.drop_target = None, None
                    
                    elif e.type == pygame.MOUSEMOTION:
                        if self.dragging_scrollbar:
                            dy = e.pos[1] - self.drag_start_y
                            if self.dragging_scrollbar == 'agent' and self.agent_list_thumb:
                                total_h = len(self.agents) * 35; view_h = HEIGHT - 150; max_scroll = total_h - view_h
                                if max_scroll > 0:
                                    scroll_ratio = (view_h - self.agent_list_thumb.height) / max_scroll
                                    self.agent_scroll = max(0, min(max_scroll, self.drag_start_scroll + dy / scroll_ratio if scroll_ratio > 0 else 0))
                            elif self.dragging_scrollbar == 'editor' and self.editor_panel_thumb:
                                total_h = 2000; max_scroll = total_h - HEIGHT
                                if max_scroll > 0:
                                    scroll_ratio = (HEIGHT - self.editor_panel_thumb.height) / max_scroll
                                    self.editor_scroll = max(0, min(max_scroll, self.drag_start_scroll + dy / scroll_ratio if scroll_ratio > 0 else 0))
                    
                    elif e.type == pygame.MOUSEWHEEL:
                         mouse_pos = pygame.mouse.get_pos(); scroll_amount = -e.y * 20
                         if mouse_pos[0] < 320: self.agent_scroll = max(0, self.agent_scroll + scroll_amount)
                         elif 340 <= mouse_pos[0] < 960 and self.editing: self.editor_scroll = max(0, self.editor_scroll + scroll_amount)
            
            # --- Drawing Logic ---
            event_for_ui = None if (self.step_modal or self.pending_deletion_agent) else click_event
            self.draw_agent_list(event_for_ui)
            if self.editing:
                self.draw_editor(event_for_ui)
            
            if self.step_modal:
                self.step_modal.draw(screen)
            elif self.pending_deletion_agent:
                self.draw_delete_confirmation()

            pygame.display.flip()
            clock.tick(60)

if __name__ == "__main__":
    Editor().run()
