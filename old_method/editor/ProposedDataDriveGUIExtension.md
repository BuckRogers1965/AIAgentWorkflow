### **Technical Specification: Agent-Driven GUI Constructs**

#### **1. Overview**

This document specifies a mechanism for creating high-level, user-friendly UI constructs (e.g., loops, conditionals) within the Workflow Editor. The design is guided by our core philosophy of a data-driven, decentralized architecture.

The intelligence and rules for these constructs will not be hard-coded into the GUI application. Instead, they will be defined within an optional data structure inside the JSON definition of a primary "head" agent. The GUI will act as a generic engine that reads these definitions and renders the appropriate interactive experience.

This approach ensures the GUI remains simple, stable, and extensible without requiring code changes to support new UI constructs.

#### **2. The `gui_construct` Data Structure**

To promote a standard agent into a "head agent" for a UI construct, an optional top-level key named `gui_construct` will be added to its JSON definition. If this key is absent, the agent is treated as a simple, standalone step.

The presence of this key signals to the GUI that this agent represents a multi-part structure with special design-time behavior.

The schema for the `gui_construct` object is as follows:

```json
"gui_construct": {
  "type": "String",                 // (Required) The rendering primitive for the GUI. E.g., "block_construct".
  "palette": {                      // (Required) How to display this construct in the tool palette.
    "category": "String",           // E.g., "Control Flow", "Error Handling".
    "display_name": "String",       // E.g., "For Loop", "If/Else".
    "icon": "String (filepath)"     // E.g., "icons/loop.svg".
  },
  "partner_agents": [               // (Required) A list of agents that are part of this construct.
    {
      "agent_name": "String",       // The name of the partner agent to instantiate. E.g., "for_loop_end".
      "link_type": "String"         // A hint for the GUI. E.g., "end_of_block", "else_clause".
    }
  ],
  "rules": {                        // (Required) The logic for wiring the construct together.
    "unique_id_name": "String",     // The base name for creating unique instance variables. E.g., "loop_id".
    "variable_scope": "String",     // "workflow" or "local". Defines where state variables are stored.
    "patch_targets": [              // (Optional) Instructions for modifying the agents at design time.
      {
        "target": "String",         // "self" or a partner's `agent_name`.
        "target_key": "String",     // The key to modify. E.g., "function_def".
        "template": "String",       // A template string with placeholders. E.g., "if results['{{id}}'] > 0: ..."
        "substitutions": ["String"] // A list of variables to substitute into the template.
      }
    ]
  }
}
```

#### **3. GUI Engine Behavior**

The GUI Editor must be implemented to act as a generic engine for these rules.

**3.1. On Startup:**

1.  Scan and parse all available agent definitions.
2.  For each agent, check for the existence of the `gui_construct` key.
3.  If `gui_construct` **exists**:
    *   Do **not** display the agent in the standard, flat list of tools.
    *   Use the `palette` object to render a single, high-level widget in the specified category with the given display name and icon. This widget represents the entire construct.
4.  If `gui_construct` **does not** exist:
    *   Render the agent as a simple, draggable step in the default "Agents" palette, unless its definition marks it as "hidden" (a convention for partner agents that should not be placed manually).

**3.2. On User Action (Adding a Construct to the Canvas):**

When a user drags a construct widget (e.g., "For Loop") from the palette onto the canvas:

1.  The GUI reads the `gui_construct` definition from the corresponding head agent (e.g., `for_loop_start.json`).
2.  **Instantiate Agents:**
    *   Instantiate the "head" agent itself.
    *   Iterate through the `partner_agents` list and instantiate each required partner agent, placing them in the correct sequence within the workflow's step list.
3.  **Generate Unique Context:**
    *   Generate a short, unique identifier for this specific instance of the construct (e.g., `_f4c1`).
    *   Prepend this ID to the name specified in `rules.unique_id_name` to create a collision-proof variable base (e.g., `loop_id_f4c1`).
4.  **Apply Patching Rules:**
    *   Iterate through the `rules.patch_targets` list.
    *   For each entry, locate the target agent instance (either the head agent itself or a partner) and the key to be modified (`target_key`).
    *   Perform a template substitution on the `template` string, replacing placeholders like `{{unique_id}}`, `{{jump_target_index}}`, etc., with the dynamically calculated values for this specific instance.
    *   Update the agent instance's definition with the patched string.
5.  **Render UI:**
    *   Apply visual styling based on the `type` key (e.g., `block_construct` would trigger indenting for all steps between the head and final partner agent).
    *   The GUI is responsible for maintaining the integrity of the construct. If the user deletes the head agent, the GUI should automatically delete all its managed partner agents.


# Technical Specification: Agent-Driven GUI Constructs

## 1. Core Philosophy

This specification implements a **self-describing visual language** where agents define their own visual behavior, connection patterns, and interaction rules. Just as our execution engine is Turing-complete through minimal primitives (step execution + index manipulation), our GUI achieves infinite extensibility through minimal rendering primitives controlled entirely by agent definitions.

**Key Principle**: The GUI is a generic rendering engine that only knows basic visual primitives. All semantic knowledge about control flow, validation rules, and visual behavior lives within the agents themselves.

## 2. Minimal GUI Primitives

The GUI engine implements only these core capabilities:
- Render agent boxes with specified styling
- Draw connections between inputs/outputs
- Apply visual grouping and indentation
- Execute agent-defined validation rules
- Trigger agent self-wiring on workflow changes

All complex behavior emerges from agent definitions, not GUI code.

## 3. The `gui_construct` Schema

To promote a standard agent into a "visual construct head agent," add this optional top-level key:

```json
"gui_construct": {
  "type": "String",                     // Rendering primitive: "block_construct", "decision_tree", "parallel_branches"
  "palette": {
    "category": "String",               // Tool palette category
    "display_name": "String",           // User-facing name
    "icon": "String",                   // Icon file path
    "description": "String"             // Tooltip description
  },
  "partner_agents": [
    {
      "agent_name": "String",           // Partner agent to instantiate
      "link_type": "String",            // Visual hint: "end_block", "else_clause", "catch_block"
      "required": "Boolean",            // Must exist for valid construct
      "auto_create": "Boolean"          // Create automatically when head is placed
    }
  ],
  "visual_rules": {
    "connection_pattern": "String",     // How this construct connects: "sequential", "conditional", "parallel"
    "styling": {
      "block_style": "String",          // "indented", "bracketed", "highlighted"
      "connector_style": "String",      // "straight", "curved", "dashed"
      "color_scheme": "String"          // Visual theme identifier
    },
    "validation_rules": [
      {
        "rule_type": "String",          // "nesting", "ordering", "uniqueness"
        "rule_def": "String",           // Python expression for validation
        "error_message": "String"       // User-facing error
      }
    ]
  },
  "wiring_rules": {
    "unique_id_base": "String",         // Base name for unique instance variables
    "auto_wire_code": "String",         // Python code executed on workflow changes
    "patch_targets": [
      {
        "target": "String",             // "self" or partner agent name
        "target_key": "String",         // Key to modify in agent definition
        "template": "String",           // Template with {{placeholders}}
        "substitutions": ["String"]     // Variables available for substitution
      }
    ]
  }
}
```

## 4. Control Flow Agent Examples

### 4.1 For Loop Construct

**Head Agent: `for_loop_start.json`**
```json
{
  "name": "for_loop_start",
  "type": "proc",
  "function_def": "if results.get('{{unique_id}}_counter', 0) >= {{max_iterations}}: results['step_index'] = {{end_step_index}}",
  "gui_construct": {
    "type": "block_construct",
    "palette": {
      "category": "Control Flow",
      "display_name": "For Loop",
      "icon": "icons/for_loop.svg",
      "description": "Repeats contained steps a specified number of times"
    },
    "partner_agents": [
      {
        "agent_name": "for_loop_end",
        "link_type": "end_block",
        "required": true,
        "auto_create": true
      }
    ],
    "visual_rules": {
      "connection_pattern": "sequential_with_backedge",
      "styling": {
        "block_style": "indented",
        "connector_style": "curved_back",
        "color_scheme": "blue_loop"
      },
      "validation_rules": [
        {
          "rule_type": "nesting",
          "rule_def": "validate_proper_block_nesting(self, workflow_steps)",
          "error_message": "Loop blocks cannot overlap - they must be fully nested or separate"
        }
      ]
    },
    "wiring_rules": {
      "unique_id_base": "loop",
      "auto_wire_code": "update_loop_wiring(self, partner_agents, workflow_steps)",
      "patch_targets": [
        {
          "target": "self",
          "target_key": "function_def",
          "template": "if results.get('{{unique_id}}_counter', 0) >= {{max_iterations}}: results['step_index'] = {{end_step_index}}",
          "substitutions": ["unique_id", "max_iterations", "end_step_index"]
        }
      ]
    }
  }
}
```

**Partner Agent: `for_loop_end.json`**
```json
{
  "name": "for_loop_end",
  "type": "proc",
  "function_def": "results['{{unique_id}}_counter'] = results.get('{{unique_id}}_counter', 0) + 1; results['step_index'] = {{start_step_index}}",
  "hidden": true
}
```

**Auto-wiring Pseudocode:**
```python
def update_loop_wiring(head_agent, partners, steps):
    # Find partner positions
    start_index = find_step_index(head_agent, steps)
    end_index = find_partner_step_index("for_loop_end", head_agent.unique_id, steps)
    
    # Update head agent to jump past end on completion
    head_agent.patch_function_def({
        "end_step_index": end_index + 1,
        "unique_id": head_agent.unique_id
    })
    
    # Update end agent to jump back to start
    end_agent = find_partner_agent("for_loop_end", head_agent.unique_id)
    end_agent.patch_function_def({
        "start_step_index": start_index,
        "unique_id": head_agent.unique_id
    })
```

### 4.2 If/Else Construct

**Head Agent: `if_condition.json`**
```json
{
  "name": "if_condition",
  "type": "proc",
  "function_def": "if not ({{condition}}): results['step_index'] = {{else_or_end_index}}",
  "gui_construct": {
    "type": "decision_tree",
    "palette": {
      "category": "Control Flow",
      "display_name": "If/Else",
      "icon": "icons/if_else.svg",
      "description": "Conditional execution with optional else clause"
    },
    "partner_agents": [
      {
        "agent_name": "else_clause",
        "link_type": "else_clause",
        "required": false,
        "auto_create": false
      },
      {
        "agent_name": "end_if",
        "link_type": "end_block",
        "required": true,
        "auto_create": true
      }
    ],
    "visual_rules": {
      "connection_pattern": "conditional_branches",
      "styling": {
        "block_style": "bracketed",
        "connector_style": "branched",
        "color_scheme": "green_conditional"
      },
      "validation_rules": [
        {
          "rule_type": "ordering",
          "rule_def": "validate_else_before_endif(self, partners, steps)",
          "error_message": "Else clause must come before end of if block"
        }
      ]
    },
    "wiring_rules": {
      "unique_id_base": "if",
      "auto_wire_code": "update_conditional_wiring(self, partner_agents, workflow_steps)"
    }
  }
}
```

### 4.3 Try/Catch Construct

**Head Agent: `try_block.json`**
```json
{
  "name": "try_block",
  "type": "proc",
  "function_def": "results['{{unique_id}}_error_handler'] = {{catch_step_index}}",
  "gui_construct": {
    "type": "error_handling_construct",
    "palette": {
      "category": "Error Handling",
      "display_name": "Try/Catch",
      "icon": "icons/try_catch.svg",
      "description": "Error handling with exception catching"
    },
    "partner_agents": [
      {
        "agent_name": "catch_block",
        "link_type": "catch_clause",
        "required": true,
        "auto_create": true
      },
      {
        "agent_name": "finally_block",
        "link_type": "finally_clause",
        "required": false,
        "auto_create": false
      },
      {
        "agent_name": "end_try",
        "link_type": "end_block",
        "required": true,
        "auto_create": true
      }
    ],
    "visual_rules": {
      "connection_pattern": "exception_flow",
      "styling": {
        "block_style": "highlighted",
        "connector_style": "dashed_error",
        "color_scheme": "red_error"
      }
    }
  }
}
```

### 4.4 Parallel Execution Construct

**Head Agent: `parallel_start.json`**
```json
{
  "name": "parallel_start",
  "type": "proc",
  "function_def": "spawn_parallel_branches({{branch_definitions}}, results)",
  "gui_construct": {
    "type": "parallel_branches",
    "palette": {
      "category": "Concurrency",
      "display_name": "Parallel Execution",
      "icon": "icons/parallel.svg",
      "description": "Execute multiple branches concurrently"
    },
    "partner_agents": [
      {
        "agent_name": "branch_separator",
        "link_type": "branch_divider",
        "required": true,
        "auto_create": true
      },
      {
        "agent_name": "parallel_join",
        "link_type": "end_block",
        "required": true,
        "auto_create": true
      }
    ],
    "visual_rules": {
      "connection_pattern": "fan_out_fan_in",
      "styling": {
        "block_style": "parallel_columns",
        "connector_style": "parallel_lines",
        "color_scheme": "purple_parallel"
      }
    }
  }
}
```

### 4.5 Switch/Case Construct

**Head Agent: `switch_statement.json`**
```json
{
  "name": "switch_statement",
  "type": "proc",
  "function_def": "switch_value = {{switch_expression}}; results['step_index'] = get_case_jump_target(switch_value, {{case_map}})",
  "gui_construct": {
    "type": "multi_branch_construct",
    "palette": {
      "category": "Control Flow",
      "display_name": "Switch/Case",
      "icon": "icons/switch.svg",
      "description": "Multi-way conditional branching"
    },
    "partner_agents": [
      {
        "agent_name": "case_clause",
        "link_type": "case_branch",
        "required": true,
        "auto_create": false
      },
      {
        "agent_name": "default_case",
        "link_type": "default_branch",
        "required": false,
        "auto_create": false
      },
      {
        "agent_name": "end_switch",
        "link_type": "end_block",
        "required": true,
        "auto_create": true
      }
    ],
    "visual_rules": {
      "connection_pattern": "multi_branch_tree",
      "styling": {
        "block_style": "tabbed_branches",
        "connector_style": "tree_branches",
        "color_scheme": "orange_switch"
      }
    }
  }
}
```

## 5. GUI Engine Implementation

### 5.1 Startup Behavior
```python
def initialize_palette():
    for agent_def in load_all_agent_definitions():
        if 'gui_construct' in agent_def:
            # Create high-level construct widget
            create_construct_widget(agent_def['gui_construct']['palette'])
        elif not agent_def.get('hidden', False):
            # Create simple agent widget
            create_simple_agent_widget(agent_def)
```

### 5.2 Construct Instantiation
```python
def instantiate_construct(construct_def, drop_position):
    unique_id = generate_unique_id()
    head_agent = instantiate_agent(construct_def['name'], unique_id)
    
    partner_instances = []
    for partner_def in construct_def['gui_construct']['partner_agents']:
        if partner_def['auto_create']:
            partner = instantiate_agent(partner_def['agent_name'], unique_id)
            partner_instances.append(partner)
    
    # Apply visual rules
    apply_visual_styling(head_agent, partner_instances, construct_def['gui_construct']['visual_rules'])
    
    # Execute initial wiring
    execute_wiring_rules(head_agent, partner_instances, construct_def['gui_construct']['wiring_rules'])
    
    return head_agent, partner_instances
```

### 5.3 Auto-Wiring on Changes
```python
def on_workflow_changed():
    for agent in get_all_head_agents():
        if hasattr(agent, 'gui_construct'):
            execute_auto_wire_code(agent)
            validate_construct_integrity(agent)
```

## 6. Benefits of This Architecture

1. **Infinite Extensibility**: New control constructs require only JSON definitions, no GUI code changes
2. **Self-Describing**: Each construct encodes its own visual behavior, validation rules, and wiring logic
3. **Consistent Philosophy**: Same data-driven approach as the execution engine
4. **Domain-Specific Languages**: Custom constructs can be created for specific problem domains
5. **Maintainable**: Bug fixes and enhancements happen in agent definitions, not scattered GUI code
6. **Composable**: Complex constructs can be built from simpler primitives

This approach creates a **self-describing visual language** where the vocabulary (agents) defines its own grammar (visual and behavioral rules), achieving the same elegant minimalism as our Turing-complete execution engine.



#### **7. Conclusion**

This specification enables a rich, user-friendly editing experience built upon our stable, minimalist core. It fully decouples the GUI's presentation logic from its application code, allowing for infinite extensibility of the UI through the creation of new agent definitions. This maintains the integrity of our data-driven philosophy across the entire platform, from runtime execution to design-time authoring.
