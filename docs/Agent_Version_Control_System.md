### **Documentation: The Agent Version Control System**

The Dynamic Agent Workflow framework includes a powerful, automated version control system designed to ensure stability, promote reusability, and maintain a clear history of changes without burdening the developer.

The entire system is managed at the **tooling layer** (within the IDE) and requires **zero changes to the core runtime engine**. This maintains a clean separation between the development environment and the production execution environment.

#### **1. The Structured Identifier: The Foundation**

The cornerstone of the versioning system is a structured, path-like identifier for every agent.

*   **Schema:** `/{namespace}/{sub-namespace/...}/{agent_name}/v{MAJOR}.{MINOR}`
*   **Example:** `/protocol/database/postgres_query/v1.1`

This schema provides three key pieces of information:
*   **Namespace (`/protocol/database/`):** Organizes agents into a logical hierarchy.
*   **Agent Name (`postgres_query`):** The clean, human-readable name for a family of agents.
*   **Version (`v1.1`):** A unique, immutable reference to a specific implementation of that agent.

Agents without this path-like structure are treated as "unversioned" and are handled with simple save/overwrite logic for backward compatibility.

#### **2. The User Interface: Simplicity and Control**

The IDE is designed to hide the complexity of the versioned identifiers while providing intuitive controls.

*   **Consolidated Agent List:** The main agent list displays only the clean **agent name** (e.g., `postgres_query`). All versions are grouped under this single entry. This keeps the list clean and easy to navigate.
*   **Powerful Search:** The search/filter bar matches against the **full hidden path** of an agent. A user can type `postgres` or `/protocol/` to find the `postgres_query` agent.
*   **Explicit Version Selection:** When a user clicks on an agent in the list that has multiple versions, a pop-up menu appears at the cursor, allowing them to choose exactly which version they want to open for editing. If only one version exists, it opens immediately.

#### **3. Workflow Stability: Pinned Dependencies**

To ensure that workflows are stable and produce repeatable results, all dependencies are **pinned**.

*   When an agent is dragged from the list into a workflow's "Steps" editor, the user is prompted to select a specific version to add (if multiple major versions exist).
*   The `agent` key in the workflow's step data stores the **full, unique, versioned identifier** (e.g., `.../postgres_query/v1.1`).
*   This means that a workflow will *always* execute against that exact version of the agent, even if newer versions are created later. This prevents unexpected behavior from upstream changes.
*   Developers can manually change the pinned version for any step at any time using the "Version" dropdown in the Step Editor.

#### **4. The "Never Overwrite" Save Logic**

The "Save Agent" button is the engine of the version control system. It is designed to be a non-destructive action that automatically builds the agent's history.

When saving an agent that has a versioned name:

1.  **No Changes:** If the agent's name and its entire definition are identical to what is already on disk, the save is cancelled, and the user is notified.
2.  **Non-Breaking Change (Minor Version Bump):** A change is considered "non-breaking" if it only modifies the internal logic (`function_def`, `prompt`), help text, or adds a *new* optional input.
    *   **Action:** The system automatically increments the **MINOR** version number (e.g., `v1.1` becomes `v1.2`).
    *   A **new agent** is created and saved to `config.json` under this new identifier. The original `v1.1` agent is left completely untouched.
3.  **Breaking Change (Major Version Bump):** A change is considered "breaking" if it modifies the agent's public interface in a way that could break existing workflows. This includes:
    *   Changing the `inputs` or `outputs` list.
    *   Removing or renaming an `optional_inputs` key.
    *   **Action:** The system automatically increments the **MAJOR** version number and resets the MINOR version to zero (e.g., `v1.2` becomes `v2.0`).
    *   A **new agent** is created and saved to `config.json` under this new identifier. The original `v1.2` agent is left untouched.

#### **5. Automatic Pruning and Retention Policy**

To prevent the `config.json` file from growing indefinitely, an automatic pruning process is tied to the save action.

*   **Policy:** The retention policy is defined in `config.json` under `gui_settings.version_retention_policy` (e.g., keep the last `3` versions).
*   **Protected Versions:** Before pruning, the system scans **every step of every workflow** in the entire configuration. Any agent version that is explicitly used as a pinned dependency is marked as "protected" and will **never be deleted**, regardless of the retention policy.
*   **Pruning Action:** After a new version is saved, the system looks at all unprotected versions within the same major version branch. If the number of these versions exceeds the retention policy, the oldest ones are automatically and safely deleted.

This combination of features provides a robust, professional-grade version control system that prioritizes stability, maintains a clear history, and requires minimal cognitive overhead from the developer.

