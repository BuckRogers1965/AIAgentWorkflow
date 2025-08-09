### A Developer's Guide to `template` Agents

### Introduction

The `template` agent is a powerful, lightweight tool for **data transformation and string manipulation**. Unlike the `proc` agent, which runs Python code, the `template` agent performs a sophisticated "find and replace" operation on a text string.

It is the ideal tool for formatting prompts for AI models, building payloads for API calls, combining pieces of text, or creating any kind of structured string output based on variables from your workflow.

### 1. The Anatomy of a `template` Agent

A `template` agent is defined by a simple JSON object in your `config.json` file. Its structure is similar to a `proc` agent but focused on its core string-based task.

```json
"my_template_agent": {
  "type": "template",
  "help": "A clear description of what this template generates.",
  "inputs": ["variable_1", "variable_2"],
  "optional_inputs": ["optional_variable_1"],
  "outputs": ["formatted_string"],
  "prompt": "This is the template string with {variable_1} and {variable_2} placeholders."
}
```

#### Key-Value Breakdown:

| Key               | Required | Description                                                                                                                              |
| :---------------- | :------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **`type`**            | Yes      | Must always be `"template"`.                                                                                                             |
| **`help`**            | Yes      | A human-readable description of the template's purpose.                                                                                  |
| **`inputs`**          | Yes      | A list of string names for variables that are **required** for this template. The engine will look for these in the current scope.       |
| **`optional_inputs`** | Yes      | A list of string names for variables that are **optional**. The template will still work if these are missing.                           |
| **`outputs`**         | Yes      | Must contain **exactly one** string name for the variable that will hold the final, formatted string.                                   |
| **`prompt`**          | Yes      | The template string itself. This is the core of the agent. It contains plain text and placeholders for your variables.                  |

---

### 2. The Core Concept: Placeholders `{...}`

The power of a `template` agent comes from its **placeholders**. A placeholder is any variable name from your `inputs` or `optional_inputs` list, wrapped in curly braces `{}`.

When the workflow engine executes a `template` agent, it reads the `prompt` string and searches for these placeholders. For each one it finds, it replaces it with the corresponding value from the workflow's current state.

**Example:**

*   **`prompt` string:** `Hello, {user_name}! Welcome to {product_name}.`
*   **`inputs`:** `["user_name", "product_name"]`
*   If the workflow has a variable `user_name` with the value `"Alice"` and a variable `product_name` with the value `"The Agent Framework"`, the final output will be:
    `"Hello, Alice! Welcome to The Agent Framework."`

---

### 3. The Power of Scoping

The `template` agent doesn't just look at the immediate inputs you provide in a workflow step. It has access to a rich **scope** of all variables that have been created up to that point in the workflow.

The engine builds this scope in a specific order of priority, with later additions overwriting earlier ones if a name collision occurs:

1.  **Previous Step `outputs`:** All variables created by all previous steps in the workflow are available.
2.  **CLI Arguments:** Any arguments provided when the workflow was started from the command line are added.
3.  **Current Step `params`:** The parameters defined for the current `template` agent step are added last, giving them the highest priority.

This layered scope is what makes templates so powerful.

**Example: A Multi-Source Template**

Imagine a workflow that takes a command-line argument, runs a `proc` agent, and then uses a `template`.

*   **CLI Call:** `python dynamic_workflows_agents.py my_workflow --customer_id "C-123"`
*   **Step 1 (proc):** An agent `get_user_data` runs and produces an output variable named `user_name` with the value `"Bob"`.
*   **Step 2 (template):**
    ```json
    // The 'append_text' agent is a template.
    {
      "agent": "append_text",
      "params": {
        "part_text": " logged in." // This defines 'part_text' in the highest-priority scope.
      },
      "prompt": "{user_name} ({customer_id}) has {part_text}",
      "output": ["log_entry"]
    }
    ```

When this `append_text` template runs, the engine builds the scope:
1.  From Step 1: `{"user_name": "Bob"}`
2.  From CLI: `{"customer_id": "C-123"}`
3.  From current step's `params`: `{"part_text": " logged in."}`

It then populates the prompt `"{user_name} ({customer_id}) has {part_text}"` using this complete scope to produce the final output:

**`log_entry`**: `"Bob (C-123) has logged in."`

### 4. Advanced: Indirect Variable Resolution (`$`)

The `template` agent also benefits from the engine's ability to resolve variables indirectly using the `$` prefix. This is most powerful within a workflow step's `params`.

```json
{
  "agent": "append_text",
  "params": {
    "whole_text": "$chapter",       // Use the current value of the 'chapter' variable
    "part_text": "$scatter_part"  // Use the current value of the 'scatter_part' variable
  },
  "output": ["chapter"]
}
```

When this step runs, the engine first resolves the `$chapter` and `$scatter_part` variables to their actual string values. These resolved values are then added to the scope and become available to the `append_text` template's `prompt` string (`{whole_text}{part_text}`).

This allows you to dynamically build up text by using the outputs of previous steps as the inputs for your templates.

### When to Use a `template` Agent

*   When you need to format a string.
*   When you need to create a prompt for an AI model.
*   When you need to build a JSON payload for an API call.
*   When you need to concatenate or merge multiple string variables.
*   When your task is purely about **data transformation**, not complex logic, calculation, or external I/O.

If you need `if/else` logic, loops, or to call an external library, you should use a `proc` agent. Otherwise, the lightweight and powerful `template` agent is the right tool for the job.