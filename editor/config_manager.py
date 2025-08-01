# config_manager.py
import json
import os

class ConfigManager:
    """ Manages loading, modifying, and saving the config.json file. """
    def __init__(self, config_path="config.json"):
        self.config_path = config_path
        self.config = {"agents": {}}
        self.load()

    def load(self):
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r') as f:
                try:
                    self.config = json.load(f)
                except json.JSONDecodeError:
                    print("Warning: config.json is corrupted. Starting fresh.")
                    self.config = {"agents": {}}
        if "agents" not in self.config:
            self.config["agents"] = {}

    def save(self):
        self.config["agents"] = {k: self.config["agents"][k] for k in sorted(self.config["agents"].keys())}
        with open(self.config_path, 'w') as f:
            json.dump(self.config, f, indent=2)

    def get_agent_names(self):
        return sorted(self.config["agents"].keys())

    def get_agent_data(self, agent_name):
        return self.config["agents"].get(agent_name)

    def update_agent(self, agent_name, agent_data):
        self.config["agents"][agent_name] = agent_data

    def rename_agent(self, old_name, new_name):
        if new_name in self.config["agents"] and new_name != old_name:
            return False
        self.config["agents"][new_name] = self.config["agents"].pop(old_name)
        # Also update any steps that were using the old name
        for agent in self.config["agents"].values():
            if agent.get("type") == "workflow":
                for step in agent.get("steps", []):
                    if step.get("agent") == old_name:
                        step["agent"] = new_name
        return True

    def delete_agent(self, agent_name):
        if agent_name in self.config["agents"]:
            del self.config["agents"][agent_name]

    def check_agent_usage(self, agent_to_check):
        dependents = []
        for agent_name, agent_data in self.config["agents"].items():
            if agent_name == agent_to_check: continue
            if agent_data.get("type") == "workflow":
                for step in agent_data.get("steps", []):
                    if step.get("agent") == agent_to_check:
                        dependents.append(agent_name); break
        return dependents

    def generate_unique_name(self, base):
        i = 1
        while f"{base}_{i}" in self.config["agents"]: i += 1
        return f"{base}_{i}"

    def create_new_agent(self, agent_type):
        name = self.generate_unique_name(f"new_{agent_type}")
        new_agent = {"type": agent_type, "help": "", "inputs": [], "optional_inputs": [], "outputs": []}
        if agent_type == "workflow": new_agent.update({"return_on_fail": 0, "steps": []})
        elif agent_type == "template": new_agent.update({"prompt": ""})
        elif agent_type == "proc": new_agent.update({"function": "", "function_def": ""})
        self.config["agents"][name] = new_agent
        return name