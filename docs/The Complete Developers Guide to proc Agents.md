
## The Complete Developer's Guide to `proc` Agents

### Introduction

Welcome to the core of the workflow engine. The `proc` (or "procedural") agent is the fundamental building block for extending the framework with custom logic. It allows you to wrap any self-contained Python function into a reusable, modular component that can be seamlessly integrated into any workflow.

This guide provides a comprehensive explanation of how to create, configure, and code `proc` agents. Mastering these concepts will allow you to build powerful and reliable automations.

### 1. The Anatomy of a `proc` Agent

A `proc` agent is defined by a JSON object within your `config.json` file. This object has two primary responsibilities: defining the agent's **interface** (how other agents talk to it) and providing its **implementation** (the Python code it runs).

```json
"my_agent_name": {
  "type": "proc",
  "help": "A clear, one-sentence description of what this agent does.",
  "inputs": ["required_arg_1", "required_arg_2"],
  "optional_inputs": ["optional_arg_1", "optional_arg_2"],
  "outputs": ["result_1", "result_2"],
  "function": "my_python_function_name",
  "function_def": "..."
}
```

#### Key-Value Breakdown:

| Key               | Required | Description                                                                                                                              |
| :---------------- | :------- | :--------------------------------------------------------------------------------------------------------------------------------------- |
| **`type`**            | Yes      | Must always be `"proc"`.                                                                                                                 |
| **`help`**            | Yes      | A human-readable description. This is crucial as it's used by the IDE, CLI, and other tools to explain the agent's purpose.               |
| **`inputs`**          | Yes      | A list of string names for parameters that are **required**. The workflow engine will raise an error if a step does not provide these.      |
| **`optional_inputs`** | Yes      | A list of string names for parameters that are **optional**. Your function must provide default values for these.                         |
| **`outputs`**         | Yes      | A list of string names for the variables this agent will create. These names are essential for the `return` statement in your code.    |
| **`function`**        | Yes      | The name of the Python function to execute. This **must** match the `def function_name(...):` inside your `function_def`.                   |
| **`function_def`**    | Yes      | A string containing the complete, self-contained Python code for your agent's logic.                                                  |

---

### 2. The Core Contract: The Python Function

The code you write in the `function_def` string is the heart of your agent. To work correctly with the workflow engine, it must adhere to a specific contract for its function signature and its return value.

#### The Function Signature: A Direct and Explicit Mapping

The engine directly maps the `inputs` and `optional_inputs` from your JSON definition to the arguments of your Python function. This mapping is explicit and based on names.

##### Required Inputs
For every parameter name listed in the `inputs` array, your function signature must have a corresponding positional argument.

##### Optional Inputs
For every parameter name listed in the `optional_inputs` array, your function signature must have a corresponding **keyword argument with a default value**. This is a strict requirement.

##### The Special `output` Argument
The engine always passes one additional, required argument to your function: `output`. This is a Python list containing the names from your `outputs` array. You will use this list to construct your result dictionary when using the multi-output pattern.

**Example Signature Mapping:**

*   **JSON Definition:**
    ```json
    "inputs": ["url"],
    "optional_inputs": ["timeout", "retries"],
    "outputs": ["content", "status_code"],
    ```

*   **Corresponding Python Function Signature:**
    ```python
    def my_api_call(url: str, output: list, timeout: int = 30, retries: int = 3) -> tuple:
        # ... function code ...
    ```

#### The Return Value: A `(result, status)` Tuple

Every `proc` function must end by returning a **tuple** containing exactly two items: the result of the operation and a status dictionary.

```python
return result_data, status_dictionary
```

**1. The `status` Dictionary (Mandatory)**
This dictionary informs the engine about the outcome of the agent's execution. It must have the following structure:

*   **On Success:**
    ```python
    {"status": {"value": 0, "reason": "Success"}}
    ```
*   **On Failure:**
    ```python
    {"status": {"value": 1, "reason": "A descriptive error message."}}
    ```
The `value` of `0` signals success, while `1` signals failure. The workflow engine uses this `value` to decide whether to continue to the next step or halt execution (if `return_on_fail` is enabled in the workflow).

**2. The `result_data` (Flexible)**
This is the data your agent produces. The framework provides two powerful patterns for returning `result_data`, depending on how many outputs your agent has.

##### **Pattern 1: The Multi-Output Pattern (Returning a Dictionary)**

This pattern is **mandatory if your agent needs to produce more than one output**. You return a Python dictionary where the keys are strings that match the names in your `outputs` list.

*   **Use this when:**
    *   You have two or more `outputs` defined (e.g., `["quotient", "remainder"]`).
    *   You want your code to be highly explicit, even with a single output.

*   **Example:**
    ```python
    # JSON: "outputs": ["quotient", "remainder"]
    
    # Python code:
    quotient = 10 // 3
    remainder = 10 % 3
    result_dictionary = {
        output[0]: quotient,  # 'quotient'
        output[1]: remainder  # 'remainder'
    }
    return result_dictionary, success_status
    ```

##### **Pattern 2: The Single-Output Shortcut (Returning a Raw Value)**

This is a convenient shortcut that **only works if your agent produces a single output**. You can return the value directly, without wrapping it in a dictionary. The engine will automatically assign this value to the first (and only) variable name listed in your `outputs`.

*   **Use this when:**
    *   You have exactly one `output` defined (e.g., `["sum"]`).
    *   Your agent's purpose is to calculate or retrieve a single piece of data.

*   **Example:**
    ```python
    # JSON: "outputs": ["sum"]
    
    # Python code:
    result = 10 + 32
    # No dictionary needed. Just return the raw value.
    return result, success_status
    ```
The engine automatically handles this, making your code cleaner for simple, single-purpose agents.

---

### 4. Best Practices

*   **Keep Agents Small and Focused:** A good agent does one thing well. Instead of a monolithic agent that does five things, build five smaller agents and chain them together in a workflow.
*   **Write Good `help` Text:** Your future self (and your teammates) will thank you. A clear `help` string makes the agent discoverable and easy to use in the IDE.
*   **Handle Errors Gracefully:** Don't let your function crash. Use `try...except` blocks to catch potential errors and return a proper failure `status` dictionary with a clear `reason`.
*   **Use Type Hinting:** While not required, adding type hints (e.g., `url: str`) to your function signature makes your code easier to read and understand.
*   **Add a Unit Test:** Use the IDE's "Run & Test" feature to create a `run_config` block for your agent. This embeds a test directly into the agent's definition, ensuring it remains reliable as the framework evolves.


Of course. This is a crucial philosophical point about the framework's design. Adding this section will elevate the guide from a technical "how-to" to an architectural "why."

Here is the extension, designed to be added as a final, high-level section to the guide.

---

### 5. The Agent as the Domain Expert

Beyond the technical implementation, it's essential to understand the core design philosophy of a `proc` agent: **The agent is the single source of truth and the domain expert for the task it performs.**

This means that all the specialized knowledge required to perform a task reliably—including error handling, retry policies, and protocol-specific behaviors—should be encapsulated *inside* the agent itself, not managed by the workflow that calls it. The workflow should remain simple, declarative, and largely ignorant of the complex inner workings of the agents it orchestrates.

#### An Agent Sets Its Own Policies

Consider an agent that communicates with an external API. This API might have specific rules:
*   It might return a `429 Too Many Requests` status code when the rate limit is exceeded.
*   It might occasionally fail with a `503 Service Unavailable` error during maintenance.
*   It might require an exponential backoff strategy for retries.

This specialized knowledge does not belong in the workflow. The workflow's job is to say, "get me the data." The agent's job is to be the **expert** on *how* to get that data reliably from that specific API.

The agent, as the domain expert, is therefore responsible for implementing its own policies.

*   **Retry Policy:** The agent knows which error codes are transient and worth retrying (`503`) and which are permanent and should cause an immediate failure (`404 Not Found`). It implements its own retry loop with appropriate delays.
*   **Error Handling:** The agent knows how to parse the specific error messages from the API and can return a clean, standardized failure `reason` to the workflow.
*   **Data Adaptation:** The agent knows the structure of the API's response and is responsible for parsing it and returning the data in a predictable format, as defined by its `outputs`.

#### Exposing Policies as Overridable Inputs

While the agent is the expert, a good expert allows for flexibility. You should expose key policy decisions as **optional inputs** with sensible defaults. This makes your agent powerful by default, but configurable for advanced use cases.

This is achieved by adding parameters to your `optional_inputs` list and the corresponding function signature.

**Example: A Configurable Retry Policy**

Let's upgrade a simple API agent to expose its retry policy.

*   **JSON Definition:**
    ```json
    "api_fetcher": {
      "help": "Fetches data from a specific API with a robust, configurable retry policy.",
      "inputs": ["endpoint"],
      "optional_inputs": ["max_retries", "initial_delay"], // <-- Exposing the policy
      "outputs": ["api_data"],
      "function": "api_fetcher",
      "function_def": "..."
    }
    ```

*   **Python Function:**
    ```python
    def api_fetcher(endpoint: str, output: list, max_retries: int = 3, initial_delay: int = 5) -> tuple:
        # 'max_retries' and 'initial_delay' have sensible defaults...
        delay = initial_delay
        for attempt in range(max_retries):
            try:
                # ... make API call ...
                if response.status_code == 200:
                    return response.json(), {"status": {"value": 0, "reason": "Success"}}
                
                # ... check for retryable status codes ...
                if is_retryable and attempt < max_retries - 1:
                    time.sleep(delay)
                    delay *= 2 # Exponential backoff
                    continue
                
                # ... handle permanent failure ...

            except Exception as e:
                # ... handle network exceptions ...

        # If the loop finishes, all retries failed
        return {}, {"status": {"value": 1, "reason": "Request failed after all retries."}}
    ```

#### How This Empowers the Developer

By designing your agents this way:

1.  **Simplicity for the 90% Use Case:** A workflow developer can simply call the agent without worrying about its retry logic.
    ```json
    { "agent": "api_fetcher", "params": { "endpoint": "/users" }, "output": ["user_list"] }
    ```

2.  **Power for the 10% Use Case:** For a particularly slow or flaky endpoint, a developer can override the default policy directly in the workflow step, without ever needing to modify the agent's code.
    ```json
    {
      "agent": "api_fetcher",
      "params": {
        "endpoint": "/reports",
        "max_retries": "5",       // <-- Overriding the policy
        "initial_delay": "10"
      },
      "output": ["report_data"]
    }
    ```

By embedding expertise and policy within your agents and exposing them as configurable inputs, you create components that are not just reusable, but also intelligent, resilient, and adaptable to the specific needs of any workflow.