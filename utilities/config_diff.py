#!/usr/bin/env python3

import json
import argparse
import sys
import os

def load_agents_from_config(filepath: str) -> dict:
    """Loads a config.json file and returns its 'agents' dictionary."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config.get('agents', {})
    except FileNotFoundError:
        print(f"Error: File not found at '{filepath}'", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: File '{filepath}' is not a valid JSON file.", file=sys.stderr)
        sys.exit(1)

def find_differing_keys(dict_a: dict, dict_b: dict) -> list:
    """Compares two agent dictionaries and returns a list of keys with different values."""
    all_keys = set(dict_a.keys()).union(set(dict_b.keys()))
    differing = []
    for key in sorted(list(all_keys)):
        val_a = dict_a.get(key)
        val_b = dict_b.get(key)
        # Use canonical JSON dumps to robustly compare any value (including lists/dicts)
        if json.dumps(val_a, sort_keys=True) != json.dumps(val_b, sort_keys=True):
            differing.append(key)
    return differing

def main():
    """Main function to compare two config files and print differences."""
    parser = argparse.ArgumentParser(
        description="Compare two config.json files and report differences in their agent definitions. \n Output chuncks of json that can be cut and paste between config files to reduced typing errors. "
    )
    parser.add_argument("file_a", help="Path to the first config.json file.")
    parser.add_argument("file_b", help="Path to the second config.json file.")
    args = parser.parse_args()

    agents_a = load_agents_from_config(args.file_a)
    agents_b = load_agents_from_config(args.file_b)

    names_a = set(agents_a.keys())
    names_b = set(agents_b.keys())

    # --- Section 1: Agents unique to the first file ---
    unique_to_a = sorted(list(names_a - names_b))
    print(f"\n--- Agents found only in first config file ---\n")
    if unique_to_a:
        for i, name in enumerate(unique_to_a):
            agent_json = json.dumps(agents_a[name], indent=2)
            comma = "," if i < len(unique_to_a) - 1 else ""
            print(f'"{name}": {agent_json}{comma}\n')
    else:
        print("(None)\n")

    # --- Section 2: Agents unique to the second file ---
    unique_to_b = sorted(list(names_b - names_a))
    print(f"\n--- Agents found only in second config file ---\n")
    if unique_to_b:
        for i, name in enumerate(unique_to_b):
            agent_json = json.dumps(agents_b[name], indent=2)
            comma = "," if i < len(unique_to_b) - 1 else ""
            print(f'"{name}": {agent_json}{comma}\n')
    else:
        print("(None)\n")

    # --- Section 3: Detailed diff for common agents with different content ---
    print(f"\n--- Agents with different content (found in both files) ---\n")
    common_names = names_a.intersection(names_b)
    found_differences = False

    for name in sorted(list(common_names)):
        agent_a = agents_a[name]
        agent_b = agents_b[name]
        
        differing_keys = find_differing_keys(agent_a, agent_b)
        
        if differing_keys:
            found_differences = True
            print(f"--- Agent '{name}' differs ---\n")
            
            for key in differing_keys:
                # Print the differing fragment from the first file
                print(f"First config file has this value for '{key}':")
                if key in agent_a:
                    value_a_json = json.dumps(agent_a[key], indent=2)
                    print(f'"{key}": {value_a_json},\n')
                else:
                    print("(key not present in first file)\n")
                
                # Print the differing fragment from the second file
                print(f"Second config file has this value for '{key}':")
                if key in agent_b:
                    value_b_json = json.dumps(agent_b[key], indent=2)
                    print(f'"{key}": {value_b_json},\n')
                else:
                    print("(key not present in second file)\n")

    if not found_differences:
        print("(None)")
    print()


if __name__ == "__main__":
    main()
