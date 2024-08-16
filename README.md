# AIAgentWorkflow
A python based AI Agent Workflow system that is dynamic and highly configurable. 

This project started after I installed the Ollama AI program and put a workflow on my local network.  I downloaded a shell based example that used curl to ask the ai api "Why is the sky blue."  This quickly turned into a python program that created agents by filling out templates using inputs from either varaibles I set, or the output from other agents. 

I noticed that I was duplicating a lot of code to generate these requests as I let the data pass through all the agents one after the other.  So I set out to fix that. I wrote a program to duplicate the simple python program in a more robust way flexible that was ran from a config file. I only had api agents at that time.  I saw where I could refactor things to make the core very light weight and flexible and super powerful by creating proc agents and duplicating the initial api agents as just a handful of agents. And I implemented dynamic function creation from the config file in just a few lines of code. Then I chopped out 500 lines of code, half the program at that time. That is when the current system came into being.  

There are just three agent types.  Templates, Procs, and Workflows. Templates are more limited than procs in function, but they have full access to all the variables in the system. I implement agents in terms of blocks of text that have parts inserted at different places.  Procs are super powerful because they are python functions, so they can do anything that python can do.  But they are limited in that they can only see what is passed into them as inputs from the step varaibles. 

Workflows also have inputs and outputs, just like templates and procs.  You can write agents and combine those agents into a workflow. The steps in a workflow map all the inputs and outputs between the steps. That workflow is an agent that you can include other agents.  Because the functions you write and put into the proc can do anything, that means you can write agents that connect to any data source and send and recieve data in the format you want.  I am already connecting to local ollama, stable diffusion, and out to cloud apis. It just works. 

Each workflow has its own results area, think of this as a scratch pad. The inputs are read from this scratch pad or the scoped variables such as from the command line options. The outputs from a step are written to this scratch pad.  If you nest a workflow inside a workflow it gets its own results area, and only the inputs to that step are available to it.  The whole system is basically just a Turing machine with nested Turing machines. 

Anything can also return one or more results that are written into the results area for use by agents later. These results can be inputs to any other step in the workflow.  The program control flow (step_index) is also just a varaible in this results area.  So that means you can implement any control flow you want.  No limits. I initially built the loop start and loop end agents as two different agent types and that is when I put the step_index into the results so the program could control its own program flow, and reimplemented the loop agents as simple proc agents that communicate using results. 

There is a status return for everything, it also just goes into the results scratch pad. The next step can read the status of the previous step and change the flow of execution.  A workflow can also terminate if any status is a fail just by setting a flag in that workflow.

One of the more powerful features is that every agent and workflow is exposed to the command line interface. It automatically requires the proper inputs to be passed in on the command line and returns the results to stdout. All the configurations and input checks are automatic. 

The latest update I made to the system is using environmental varaibles for secrets in the config file that are only seen by the interals of the proc agent. And I created optional parameters for a few new cloud api agents. 

This program came together in just a month's time. I was amazed at how fast the program evolved.  

---

To show you how simple these configs are:

This is a proc agent with embeded code:

    "read_file": {
            "type": "proc",
            "help": "Reads content from a file",
            "function": "read_file",
            "function_def": "def read_file(file_name: str,output:list) -> bytes:\n\twith open(file_name, 'rb') as f:\n\t\treturn f.read(), {\"status\": {\"value\": 0, \"reason\": \"Success\"}}\n",
            "inputs": [ "file_name" ],
            "optional_inputs": [],
            "outputs": [ "file_content" ]
        },

This is a template:

    "echo": {
            "type": "template",
            "help": "echo input to output",
            "prompt": "{input}",
            "inputs": [ "input" ],
            "optional_inputs": [],
            "outputs": [ "output" ]
        }

This is a simple workflow, it gets an idea and turns it into a stable diffusion prompt using text completion, and has stable diffusion create an image, saving a png file:

        "image_agent_wf": {
            "type": "workflow",
            "help": "Generate a prompt and create an image.",
            "inputs": [
                "topic",
                "negative_prompt",
                "filename"
            ],
            "optional_inputs": [],
            "outputs": [
                "success"
            ],
            "prompt": "{prompt}",
            "steps": [
                {
                    "agent": "image_prompt_generator",
                    "params": {
                        "topic": "$topic"
                    },
                    "output": [ "image_prompt" ]
                },
                {
                    "agent": "get_ollama_response",
                    "params": {
                        "prompt": "$image_prompt"
                    },
                    "output": [ "image_prompts" ]
                },
                {
                    "agent": "get_sd_response",
                    "params": {
                        "prompt": "$image_prompts",
                        "negative_prompt ": "$negative_prompt"
                    },
                    "output": [ "base64_image" ]
                },
                {
                    "agent": "decode_base64",
                    "params": {
                        "data": "$base64_image"
                    },
                    "output": [ "decoded_data" ]
                },
                {
                    "agent": "write_file",
                    "params": {
                        "file_name": "$filename",
                        "file_content": "$decoded_data"
                    },
                    "output": ["success" ]
                }
            ]
        },

One of the things that really helps when building workflows is to create smaller workflows and test them, then combine those into higher level workflows. In the above workflow example, the get ollama response and get sd response are both nested workflows.  

To Do: 

-GUI
-Sandbox the dynamic functions to ensure they are safe.
-Set it up so the program can accept inputs from multiple config files, so you can have a system config file, and a config file with works in progress. 
-Collaborate with others to build out 1000's of agents to do everything and connect to everything.




