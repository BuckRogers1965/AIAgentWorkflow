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

#### **4. Example: `if_block` Construct**

The head agent `if_condition` would define the entire `if/else` structure.

**`if_condition.json`:**
```json
{
  "name": "if_condition",
  "type": "proc",
  "function_def": "if not {{condition}}: results['step_index'] = {{jump_to_else_or_end}}",
  "gui_construct": {
    "type": "block_construct_with_clauses",
    "palette": { "category": "Control Flow", "display_name": "If/Else Block", "icon": "icons/if.svg" },
    "partner_agents": [
      { "agent_name": "else_clause", "link_type": "else_clause" },
      { "agent_name": "end_if", "link_type": "end_of_block" }
    ],
    "rules": {
      "unique_id_name": "if_id",
      // ... patching rules to manage jumps to the 'else' block and the 'end_if' ...
    }
  }
}
```

#### **5. Conclusion**

This specification enables a rich, user-friendly editing experience built upon our stable, minimalist core. It fully decouples the GUI's presentation logic from its application code, allowing for infinite extensibility of the UI through the creation of new agent definitions. This maintains the integrity of our data-driven philosophy across the entire platform, from runtime execution to design-time authoring.
