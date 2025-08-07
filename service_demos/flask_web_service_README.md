# Dynamic Agent Workflow - Web Service Wrapper Using flask

This document provides instructions for running and interacting with the flask-based web service for the Dynamic Agent Workflow engine.

This service acts as a simple, powerful wrapper that exposes the agents defined in your `config.json` as web API endpoints. It allows you to execute complex workflows via simple HTTP requests.

## Table of Contents
1.  [Prerequisites](#1-prerequisites)
2.  [Running the Service](#2-running-the-service)
    *   [Full Access Mode (Development)](#full-access-mode-development)
    *   [Filtered Contract Mode (Production)](#filtered-contract-mode-production)
3.  [Interacting with the Service](#3-interacting-with-the-service)
    *   [Discovering Available Agents (Getting the API Contract)](#discovering-available-agents-getting-the-api-contract)
    *   [Executing an Agent](#executing-an-agent)
4.  [Troubleshooting: Firewall Configuration](#4-troubleshooting-firewall-configuration)
    *   [Linux (using ufw)](#linux-using-ufw)
    *   [Windows (using PowerShell)](#windows-using-powershell)
    *   [macOS (using pf)](#macos-using-pf)

---

## 1. Prerequisites

Before running the service, ensure you have the following:

*   **Python 3.10+** installed.
*   The required Python libraries. Install them using pip:
    ```bash
    pip install flask
    ```
*   Your core framework files:
    *   `dynamic_workflows_agents.py` (The core engine)
    *   `config.json` (Your agent and service contract definitions)
    *   `flask_web_service.py` (This web service wrapper)

## 2. Running the Service

The web service is started from the command line. The two most important arguments are `--config` to point to your agent definitions and `--lib-path` to point to the directory containing the core engine.

### Full Access Mode (Development)

This mode exposes **all** agents defined in your `config.json`. It is ideal for local development and testing. To run in this mode, simply omit the `--service` flag.

```bash
# Assuming flask_web_service.py is in a subdirectory like 'server/'
# and your core files are in the parent directory.
python flask_web_service.py --config ../config.json --lib-path ..
```

The server will start, typically on port `5000`.

### Filtered Contract Mode (Production)

This mode is for production or public-facing deployments. It exposes **only** the agents that have a specific tag in their `"web_services"` list. This creates a clean, secure, and limited API surface.

To use this mode, provide the `--service` flag with the name of the contract you want to activate.

```bash
# Activate the contract named 'public_api'
python flask_web_service.py --config ../config.json --lib-path .. --service public_api

# Activate a different contract for internal tools
python flask_web_service.py --config ../config.json --lib-path .. --service internal_tools --port 5001
```

---

## 3. Interacting with the Service

You can interact with the service using a web browser, `curl`, or any HTTP client. All responses are in XML format.

### Discovering Available Agents (Getting the API Contract)

To see which agents are available under the currently active contract, make a `GET` request to the root URL of the service.

**Request:**
```bash
curl http://127.0.0.1:5000/
```

**Example Response (`--service public_api` active):**
The response will be an XML document listing all exposed agents and their required inputs, optional inputs, and outputs.

```xml
<?xml version="1.0" ?>
<availableAgents serviceName="Dynamic Agent Workflow Service" activeContract="public_api">
  <agent>
    <name>google_search</name>
    <help>Do google search, analyze the results.</help>
    <inputs>
      <param>request</param>
    </inputs>
    <optionalInputs/>
    <outputs>
      <param>response</param>
    </outputs>
  </agent>
  <agent>
    <name>read_file</name>
    <help>Reads content from a file</help>
    <inputs>
      <param>file_name</param>
    </inputs>
    <optionalInputs/>
    <outputs>
      <param>file_content</param>
    </outputs>
  </agent>
</availableAgents>
```

### Executing an Agent

To execute an agent, construct a URL by providing the agent's name in the path and its parameters in the query string.

**URL Format:**
`http://<host>:<port>/execute/<agent_name>?<param1>=<value1>&<param2>=<value2>`

**Example: Calling the `read_file` agent**

1.  **From Discovery:** We know the agent is named `read_file` and requires one input: `file_name`.
2.  **Construct the URL:**
    `http://127.0.0.1:5000/execute/read_file?file_name=hello.world`
3.  **Make the Request:**
    ```bash
    # Note: curl will automatically URL-encode special characters if needed.
    curl "http://127.0.0.1:5000/execute/read_file?file_name=hello.world"
    ```

**Example Response:**
The service will execute the agent and return a detailed XML response containing the status, the results, and a full execution log within a `CDATA` block.

```xml
<?xml version="1.0" ?>
<agentResponse>
  <agent>read_file</agent>
  <status>
    <value>0</value>
    <reason>Success</reason>
  </status>
  <results>
    <file_content>Hello, World!!!</file_content>
  </results>
  <log>
    <![CDATA[ 2025-08-08 10:30:00,123 [INFO] Executing workflow at depth 1 ... ]]>
  </log>
</agentResponse>
```

---

## 4. Troubleshooting: Firewall Configuration

If you are running the service and cannot connect to it from another machine (or even from your local machine if the firewall is very strict), you may need to add a firewall rule to allow incoming connections on the port the service is using (default is `5000`).

Here are common commands for different operating systems. **Run these with administrator/sudo privileges.**

### Linux (using ufw)

`ufw` (Uncomplicated Firewall) is common on Ubuntu and other Debian-based systems.

```bash
# Allow incoming TCP traffic on port 5000
sudo ufw allow 5000/tcp

# Reload ufw to apply the changes
sudo ufw reload

# To check the status
sudo ufw status
```

### Windows (using PowerShell)

Run these commands in a PowerShell terminal with Administrator privileges.

```powershell
# Create a new firewall rule to allow incoming TCP traffic on port 5000
New-NetFirewallRule -DisplayName "Agent Service Port" -Direction Inbound -Protocol TCP -LocalPort 5000 -Action Allow

# To check if the rule was added, you can run:
Get-NetFirewallRule -DisplayName "Agent Service Port"
```

### macOS (using pf)

The built-in firewall `pf` on macOS is less commonly modified directly. The easiest way is often through the GUI:
1.  Go to **System Settings** > **Network** > **Firewall**.
2.  Make sure the Firewall is **On**.
3.  Click **Options...**.
4.  Click the **+** button to add a new rule.
5.  Select the `python` or `python3` application you are using to run the service and set it to "Allow incoming connections".

If you must use the command line:
1.  Create a file, e.g., `/etc/pf.anchors/my-app`, with the rule:
    ```
    pass in proto tcp from any to any port 5000
    ```
2.  Edit `/etc/pf.conf` to load this anchor rule.
3.  Reload the firewall rules with `sudo pfctl -f /etc/pf.conf`.

**Note:** Modifying system firewalls can have security implications. Only open ports that you need, and consider restricting access to trusted IP addresses if possible.
