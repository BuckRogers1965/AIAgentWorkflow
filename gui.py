import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Set, Tuple, Optional
import math

class Port:
    def __init__(self, name: str, x: int, y: int, is_input: bool, parent_node: 'Node'):
        self.name = name
        self.x = x
        self.y = y
        self.is_input = is_input
        self.parent_node = parent_node
        self.connections: Set['Connection'] = set()
        
    def draw(self, canvas: tk.Canvas):
        x = self.parent_node.x + self.x
        y = self.parent_node.y + self.y
        color = "red" if self.is_input else "green"
        canvas.create_oval(x-5, y-5, x+5, y+5, fill=color, tags="port")
        canvas.create_text(x + (-30 if self.is_input else 30), y, 
                         text=self.name, anchor="w" if self.is_input else "e")

class Connection:
    def __init__(self, start_port: Port, end_port: Port):
        self.start_port = start_port
        self.end_port = end_port
        self.id = None  # Canvas line ID
        
    def draw(self, canvas: tk.Canvas):
        start_x = self.start_port.parent_node.x + self.start_port.x
        start_y = self.start_port.parent_node.y + self.start_port.y
        end_x = self.end_port.parent_node.x + self.end_port.x
        end_y = self.end_port.parent_node.y + self.end_port.y
        
        # Create curved line using bezier curve
        ctrl_x1 = start_x + (end_x - start_x) / 3
        ctrl_x2 = end_x - (end_x - start_x) / 3
        
        self.id = canvas.create_line(
            start_x, start_y, ctrl_x1, start_y, ctrl_x2, end_y, end_x, end_y,
            smooth=True, fill="gray", width=2, tags="connection"
        )

class Node:
    def __init__(self, name: str, component_type: str, x: int, y: int):
        self.name = name
        self.component_type = component_type
        self.x = x
        self.y = y
        self.width = 200
        self.height = 150
        self.input_ports: Dict[str, Port] = {}
        self.output_ports: Dict[str, Port] = {}
        self.being_dragged = False
        
    def add_input_port(self, name: str):
        port_count = len(self.input_ports)
        y_offset = 30 + port_count * 20
        self.input_ports[name] = Port(name, 0, y_offset, True, self)
        
    def add_output_port(self, name: str):
        port_count = len(self.output_ports)
        y_offset = 30 + port_count * 20
        self.output_ports[name] = Port(name, self.width, y_offset, False, self)
        
    def draw(self, canvas: tk.Canvas):
        # Draw node rectangle
        canvas.create_rectangle(
            self.x, self.y, 
            self.x + self.width, self.y + self.height,
            fill="lightgray", tags="node"
        )
        
        # Draw title
        canvas.create_text(
            self.x + self.width/2, self.y + 15,
            text=f"{self.name}\n({self.component_type})",
            anchor="center", tags="node"
        )
        
        # Draw ports
        for port in self.input_ports.values():
            port.draw(canvas)
        for port in self.output_ports.values():
            port.draw(canvas)

class WorkflowCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(master, **kwargs)
        self.nodes: Dict[str, Node] = {}
        self.connections: List[Connection] = []
        self.selected_port = None
        self.components = {}  # Loaded component definitions
        
        self.bind("<Button-1>", self.on_click)
        self.bind("<B1-Motion>", self.on_drag)
        self.bind("<ButtonRelease-1>", self.on_release)
        
    def load_components(self, filename: str):
        with open(filename, 'r') as f:
            self.components = json.load(f)
    
    def add_node(self, name: str, component_type: str, x: int, y: int):
        node = Node(name, component_type, x, y)
        
        # Add ports based on component definition
        if component_type in self.components:
            component = self.components[component_type]
            for input_name in component.get('inputs', []):
                node.add_input_port(input_name)
            for output_name in component.get('outputs', []):
                node.add_output_port(output_name)
                
        self.nodes[name] = node
        self.draw_workflow()
        
    def draw_workflow(self):
        self.delete("all")  # Clear canvas
        
        # Draw connections first (so they appear behind nodes)
        for conn in self.connections:
            conn.draw(self)
            
        # Draw nodes
        for node in self.nodes.values():
            node.draw(self)
    
    def find_port_at(self, x: int, y: int) -> Optional[Port]:
        for node in self.nodes.values():
            for port in list(node.input_ports.values()) + list(node.output_ports.values()):
                port_x = node.x + port.x
                port_y = node.y + port.y
                if math.dist((x, y), (port_x, port_y)) < 10:
                    return port
        return None
    
    def find_node_at(self, x: int, y: int) -> Optional[Node]:
        for node in self.nodes.values():
            if (node.x <= x <= node.x + node.width and 
                node.y <= y <= node.y + node.height):
                return node
        return None
    
    def on_click(self, event):
        port = self.find_port_at(event.x, event.y)
        if port:
            self.selected_port = port
        else:
            node = self.find_node_at(event.x, event.y)
            if node:
                node.being_dragged = True
                node.drag_start_x = event.x - node.x
                node.drag_start_y = event.y - node.y
    
    def on_drag(self, event):
        if self.selected_port:
            self.draw_workflow()
            # Draw temporary line
            start_x = self.selected_port.parent_node.x + self.selected_port.x
            start_y = self.selected_port.parent_node.y + self.selected_port.y
            self.create_line(start_x, start_y, event.x, event.y, 
                           fill="gray", dash=(4, 4))
        else:
            for node in self.nodes.values():
                if node.being_dragged:
                    node.x = event.x - node.drag_start_x
                    node.y = event.y - node.drag_start_y
                    self.draw_workflow()
    
    def on_release(self, event):
        if self.selected_port:
            end_port = self.find_port_at(event.x, event.y)
            if end_port and end_port.is_input != self.selected_port.is_input:
                # Create connection
                start_port = self.selected_port if not self.selected_port.is_input else end_port
                end_port = end_port if end_port.is_input else self.selected_port
                conn = Connection(start_port, end_port)
                self.connections.append(conn)
                start_port.connections.add(conn)
                end_port.connections.add(conn)
            self.selected_port = None
            self.draw_workflow()
        
        for node in self.nodes.values():
            node.being_dragged = False

class WorkflowBuilder(tk.Tk):
    def __init__(self):
        super().__init__()
        
        self.title("Visual Workflow Builder")
        self.geometry("1200x800")
        
        # Create main container
        self.main_container = ttk.PanedWindow(self, orient=tk.HORIZONTAL)
        self.main_container.pack(fill=tk.BOTH, expand=True)
        
        # Create component list frame
        self.component_frame = ttk.Frame(self.main_container)
        self.main_container.add(self.component_frame)
        
        # Create component list
        self.component_list = ttk.Treeview(self.component_frame)
        self.component_list.pack(fill=tk.BOTH, expand=True)
        self.component_list.bind("<Double-1>", self.add_component)
        
        # Create canvas
        self.canvas = WorkflowCanvas(self.main_container, bg="white")
        self.main_container.add(self.canvas)
        
        # Create menu
        self.create_menu()
        
    def create_menu(self):
        menubar = tk.Menu(self)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Load Components", command=self.load_components)
        file_menu.add_command(label="Save Workflow", command=self.save_workflow)
        file_menu.add_command(label="Load Workflow", command=self.load_workflow)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)
        
        menubar.add_cascade(label="File", menu=file_menu)
        self.config(menu=menubar)
    
    def load_components(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            self.canvas.load_components(filename)
            self.update_component_list()
    
    def update_component_list(self):
        self.component_list.delete(*self.component_list.get_children())
        for name, component in self.canvas.components.items():
            self.component_list.insert("", "end", text=name)
    
    def add_component(self, event):
        item = self.component_list.selection()[0]
        component_type = self.component_list.item(item)["text"]
        
        # Create a dialog to get the node name
        name = tk.simpledialog.askstring(
            "Node Name",
            f"Enter name for new {component_type} node:",
            parent=self
        )
        
        if name:
            # Add node at a default position
            self.canvas.add_node(name, component_type, 100, 100)
    
    def save_workflow(self):
        filename = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            workflow = {
                "nodes": {},
                "connections": []
            }
            
            # Save nodes
            for name, node in self.canvas.nodes.items():
                workflow["nodes"][name] = {
                    "type": node.component_type,
                    "position": {"x": node.x, "y": node.y}
                }
            
            # Save connections
            for conn in self.canvas.connections:
                workflow["connections"].append({
                    "from_node": conn.start_port.parent_node.name,
                    "from_port": conn.start_port.name,
                    "to_node": conn.end_port.parent_node.name,
                    "to_port": conn.end_port.name
                })
            
            with open(filename, 'w') as f:
                json.dump(workflow, f, indent=4)
    
    def load_workflow(self):
        filename = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
        )
        if filename:
            with open(filename, 'r') as f:
                workflow = json.load(f)
            
            # Clear current workflow
            self.canvas.nodes.clear()
            self.canvas.connections.clear()
            
            # Create nodes
            for name, node_data in workflow["nodes"].items():
                self.canvas.add_node(
                    name,
                    node_data["type"],
                    node_data["position"]["x"],
                    node_data["position"]["y"]
                )
            
            # Create connections
            for conn_data in workflow["connections"]:
                start_node = self.canvas.nodes[conn_data["from_node"]]
                end_node = self.canvas.nodes[conn_data["to_node"]]
                
                start_port = start_node.output_ports[conn_data["from_port"]]
                end_port = end_node.input_ports[conn_data["to_port"]]
                
                conn = Connection(start_port, end_port)
                self.canvas.connections.append(conn)
                start_port.connections.add(conn)
                end_port.connections.add(conn)
            
            self.canvas.draw_workflow()

if __name__ == "__main__":
    app = WorkflowBuilder()
    app.mainloop()