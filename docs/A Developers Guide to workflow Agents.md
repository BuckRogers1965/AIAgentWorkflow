### A Developer's Guide to `workflow` Agents

### Introduction

The `workflow` agent is the most powerful and important component in the entire framework. It is not an atomic unit of action like a `proc` or `template`; instead, it is a **composite agent**—a container that orchestrates other agents in a sequence.

The `workflow` agent embodies the framework's core principle of **fractal self-similarity**. While it contains a sequence of steps, it presents itself to the outside world with the exact same `inputs -> outputs` interface as a simple `proc` or `template`. This allows you to build immensely complex logic and then encapsulate it into a single, clean, reusable component that can itself be a step in another, larger workflow.

This is how the system achieves Turing completeness and infinite composability.

### 1. The Anatomy of a `workflow` Agent

A `workflow` agent's definition contains the standard interface plus a `steps` array, which is the heart of its logic.

```json
"my_workflow_agent": {
  "type": "workflow",
  "help": "A clear description of what this entire sequence of steps accomplishes.",
  "inputs": ["workflow_input_1", "workflow_input_2"],
  "optional_inputs": ["optional_workflow_input_1"],
  "outputs": ["final_result_1", "final_result_2"],
  "return_on_fail": 1,
  "steps": [
    {
      "agent": "agent_name_for_step_1",
      "params": { ... },
      "output": [ ... ]
    },
    {
      "agent": "agent_name_for_step_2",
      "params": { ... },
      "output": [ ... ]
    }
  ]
}
```

#### Key-Value Breakdown:

| Key                 | Required | Description                                                                                                                                                                                          |
| :------------------ | :------- | :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`type`**              | Yes      | Must always be `"workflow"`.                                                                                                                                                                         |
| **`help`**, **`inputs`**, **`optional_inputs`**, **`outputs`** | Yes      | These define the **public interface of the entire workflow**. They act as a "black box" boundary, hiding the internal complexity.                                                     |
| **`return_on_fail`**    | No       | If set to `1` (or `true`), the workflow will halt immediately if any step returns a failure status. If `0` (or `false`, the default), it will continue executing subsequent steps.                    |
| **`steps`**             | Yes      | An array of objects, where each object represents a single step in the sequence. This is the "code" of the workflow.                                                                                  |

#### The Structure of a Step:

Each object in the `steps` array has three keys:
*   **`agent`**: The name of the agent to execute for this step.
*   **`params`**: A dictionary defining the inputs for this step's agent.
*   **`output`**: A list of string names for the new variables this step will create.

---

### 2. The Golden Rule of Workflows: Data Flow

A workflow is fundamentally a graph of data dependencies. The core concept you must master is how data flows from one step to the next.

#### Wiring Inputs: The `$` Prefix

The `params` of a step define how it gets its data. The values in the `params` dictionary can be a literal string, but more often, they are **references to existing variables** using the `$` prefix.

A variable referenced with `$` **must** have been created by one of two sources:

1.  **A Workflow Input:** It was passed into the workflow from the outside.
2.  **A Previous Step's Output:** It was created by an agent that ran earlier in the `steps` array.

**Example: Valid Data Flow**
```json
// This workflow takes 'source_file' as an input.
"inputs": ["source_file"],
"steps": [
  {
    "agent": "read_file",
    "params": {
      "file_name": "$source_file"  // VALID: Uses a workflow input.
    },
    "output": ["file_content"]
  },
  {
    "agent": "encode_base64",
    "params": {
      "data": "$file_content"    // VALID: Uses the output of the previous step.
    },
    "output": ["encoded_data"]
  }
]
```

If you try to reference a variable that doesn't exist yet (e.g., using `$encoded_data` in the first step), the workflow will likely fail or produce unexpected results. The IDE's Step Editor is designed to help you with this by showing you a list of all valid, available variables at each step.

#### Wiring Outputs: The Final Result

The `outputs` of the workflow agent itself are not automatically produced. You must ensure that a variable with that **exact name** is created by one of the steps inside the workflow.

The engine takes the final state of all variables at the end of the last step and uses it to populate the workflow's public outputs.

**Example: Valid Workflow Output**
```json
// This workflow promises to produce a variable named 'final_essay'.
"outputs": ["final_essay"],
"steps": [
  // ... many steps ...
  {
    "agent": "get_ollama_response",
    "params": { ... },
    "output": ["final_essay"] // VALID: The last step creates the promised output.
  }
]
```

If the `steps` array completes and no agent has created a variable named `final_essay`, the workflow's output will be invalid or empty. The responsibility is on you, the developer, to ensure the internal steps fulfill the public contract defined by the workflow's `outputs`.

---

### 3. The Power of Encapsulation and Recursion

The true power of this system is that once you've built and tested a workflow like `essay_generation`, you can treat it as a single, atomic unit in another, even higher-level workflow.

**Example: A Meta-Workflow**
```json
"generate_and_publish_essay": {
  "type": "workflow",
  "inputs": ["topic", "author_name"],
  "outputs": ["publish_url"],
  "steps": [
    {
      "agent": "essay_generation", // <-- Calling our previous workflow as if it were a single proc.
      "params": {
        "topic": "$topic",
        "depth": "in-depth",
        "thesis": "The impact of AI on modern writing.",
        "tone": "academic"
      },
      "output": ["generated_essay_content"]
    },
    {
      "agent": "sftp_put", // <-- Using another agent to publish the result.
      "params": {
        "hostname": "blog.example.com",
        "username": "ENV_BLOG_USER",
        "password": "ENV_BLOG_PASSWORD",
        "remote_path": "/var/www/html/posts/new_essay.html",
        "file_bytes": "$generated_essay_content"
      },
      "output": ["upload_status"]
    }
  ]
}
```
This is the fractal self-similarity in action. The `essay_generation` workflow, with all its internal complexity, is called with a simple, clean interface, just like any other agent. This allows you to build incredibly sophisticated systems by composing layers of abstraction, managing complexity at every level.

---

### 4. The IDE: Your Intelligent Assistant for Workflows

While you can write workflows by hand in a text editor, the true power and velocity of the framework are unlocked when using the visual IDE. The IDE is not just a text editor; it's an intelligent assistant that understands the framework's rules and helps you build valid, robust workflows.

#### The Step Editor: Live Variable Scoping

The most common source of errors in any workflow system is incorrectly wiring data between steps. The IDE's Step Editor is designed to eliminate this problem entirely.

When you edit a step in a workflow, the editor performs a live analysis of the entire workflow up to that point. It then presents you with a categorized list of **all valid, available variables** that can be used as inputs (`$` references) for the current step.

This list is typically broken down into:

*   **Workflow Inputs:** A list of all variables that were passed into the workflow from the outside. These are available to every step.
*   **Step Outputs:** A grouped list of all variables created by each preceding step. For example, it will show you the outputs of "Step 0: read_file," then "Step 1: encode_base64," and so on.

This feature transforms workflow development:
*   **From:** A tedious process of manually tracking variable names and scrolling up and down the file.
*   **To:** A simple, guided process of selecting from a list of known-good variables.

You no longer need to remember if a variable was named `file_content` or `file_contents`. The IDE knows and presents you with the correct option, dramatically reducing bugs and speeding up development.

#### "Promote to Workflow": The Power of Self-Similarity in Action

The fractal self-similarity of the framework—where any sequence of agents can be treated as a single agent—enables the IDE's most powerful refactoring tool: **Promote to Workflow**.

You can select any contiguous block of steps within your workflow. The IDE can treat this block as a self-contained unit because, like any agent, it has a well-defined boundary with data flowing in and out.

When you select "Promote to Workflow," the IDE automatically performs a static analysis of the selected block to determine its public interface:

1.  **It calculates the `inputs`:** It scans all the `$` variables used inside the block that are defined *outside* the block (either as workflow inputs or as outputs of steps before the block). These become the `inputs` for the new, promoted workflow.
2.  **It calculates the `outputs`:** It scans for all variables created *inside* the block that are used by steps *after* the block. These become the `outputs` of the new workflow.

The IDE then performs the refactoring automatically:
*   It creates a new `workflow` agent with the calculated `inputs` and `outputs`.
*   It moves the selected steps into this new agent.
*   It replaces the original block of steps in your current workflow with a **single step** that calls the new agent, correctly wiring up the `params` and `output` to match the interface it just calculated.

**Example:**
If you select these two steps:
```json
// ... preceding step creates $source_file ...
{ "agent": "read_file", "params": { "file_name": "$source_file" }, "output": ["file_content"] },
{ "agent": "encode_base64", "params": { "data": "$file_content" }, "output": ["encoded_data"] }
// ... subsequent step uses $encoded_data ...
```
The IDE will determine:
*   **Input:** `$source_file` (comes from outside)
*   **Output:** `$encoded_data` (is used outside)

It will then create a new `promoted_workflow_123` agent and replace the two steps with:
```json
{
  "agent": "promoted_workflow_123",
  "params": { "source_file": "$source_file" },
  "output": ["encoded_data"]
}
```

This feature is a direct result of the framework's clean, fractal architecture. It allows you to start by building a long, linear workflow and then, as you identify logical groupings, progressively refactor it into clean, reusable, encapsulated components without any manual effort.