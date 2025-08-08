## Quick Start Guide

[This project](https://github.com/BuckRogers1965/AIAgentWorkflow) is a simple, easy to understand agent workflow framework where the complexity emerges from simplicty.

The basic guiding principle is that everything is a process and those processes have inputs, perform an action on those inputs, and generates one or more outputs.  This self similarity at every level is what allows everything to grow fractally.

The 4 main parts of this tool all revolve around the config.json file.  This is the ground truth for every agent, self contained in its defintion.  This names the agent, gives it what type and behaviors it has, defines its gui behaviors, services it is a member of, and its unit test.  

There are just three agent types: workflow, template, and proc.  The workflow is just a contain for agents, but once you build a workflow it is itself an agent. This self similarity allows you to build any level of complexity of processing you need.

---

Before you run any of this project be sure to configure your system to the tools.

```bash
pip install -r requirements.txt
```
Let me know if I missed a required python module.

---

To just run the command line tool,

```bash
python dynamic_workflows_agents.py
```

This gives a list of available agents and their purpose, their inputs and outputs. 


To run one agent just give it the agent names and the inputs on the command :

```bash
python dynamic_workflows_agents.py append_text --whole_text 11111 --part_text 2222
```

```text
The output:
whole_text: 111112222
```

---

The  GUI is what moves this project into a professional level.  

What lead to creating this tool is the fact that hand editing a thousand line json file is a special kind of hell.  A single comma missing any everything collapses and breaks and it takes a long time to fix.  You had to consantly jump back and forth between the agent defintion and the step in a workflow to figure out how to wire the outputs to the inputs. If you renamed an agent, you had to find and edite that same name in every step.  The GUI solves every one of those pain points and allows direct testing as you build in the gui.

The GUI is defined by the json config structure and it just has features added to make editing that file easier. If you don't give it the path to the core library, the only thing that happens is the run button is deactivated.  It informs you of that fact.

What the gui does is give full access to the complete json config for every agent type. It gives you the tools you need to build new agents and workflows.  The step editor shows you all allows ouputs that can be mapped into the inputs of a step.  

```bash
cd editor_new_theme/
python editor_app.py --config ../config.json --lib-path ..
```

---

To generate a beautiful, well formatted report of the agents that have a defined unit test:


```bash
cd ../unit_testing/
python test_runner.py --config ../config.json --lib-path .. 
```

---

Finally, you can expose any agents you want as a web service, and you can limit direct access to any individual agents with a service tag that defines any set of interfaces you want. 


```bash
cd ../service_demo/
python flask_web_service.py --config ../config.json --lib-path ..  --service public_api 
```

Make sure that web service port is not being blocked on your system.

This allows you to show the interface as an url with

```url
http://127.0.0.1:5000/
```


You can run the same demo with: 

```url
http://127.0.0.1:5000/execute/append_text?whole_text=11111&part_text=22222
```

The result in a web page looks like:

```xml
<agentResponse>
    <script id="WXkqOoBd.js"/>
    <agent>append_text</agent>
    <status>
        <value>0</value>
        <reason>Success</reason>
    </status>
    <results>
        <whole_text>1111122222</whole_text>
    </results>
    <log>
        <![CDATA[ 2025-08-08 16:38:13,956 [INFO] Creating temporary workflow agent for agent: append_text 2025-08-08 16:38:13,956 [INFO] Executing workflow at depth 1 2025-08-08 16:38:13,956 [INFO] Starting workflow validation 2025-08-08 16:38:13,956 [INFO] Workflow validation completed successfully and has been blessed. 2025-08-08 16:38:13,956 [INFO] Executing step: append_text, type : template 2025-08-08 16:38:13,957 [INFO] Step completed. 'append_text' in 191.1959 microseconds ]]>
    </log>
</agentResponse>
```
