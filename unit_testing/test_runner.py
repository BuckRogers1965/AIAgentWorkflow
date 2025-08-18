# --- START OF FILE test_runner.py ---

import json
import argparse
import logging
import io
import re
import base64
import time
import operator
from functools import reduce
import copy
import sys
import os

# --- COMMAND-LINE ARGUMENT PARSING & VALIDATION ---
parser = argparse.ArgumentParser(
    description="Agent Workflow Test Runner. Scans config.json for agents with saved unit tests and executes them, generating an HTML report.",
    formatter_class=argparse.RawTextHelpFormatter
)
parser.add_argument(
    '--config',
    default='config.json',
    help='Path to the configuration file (default: config.json)'
)
parser.add_argument(
    '--lib-path',
    help='Path to the directory containing the dynamic_workflows_agents.py core library.'
)
parser.add_argument(
    '--service',
    help='Only run tests for agents that have this specific service contract tag.'
)
parser.add_argument(
    '--loglevel',
    default='INFO',
    choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
    help="""Set the logging level for capturing logs on FAILED tests.
- DEBUG: Most verbose, shows all steps.
- INFO: Shows standard execution flow (default).
- WARNING: Shows only warnings and errors.
- ERROR: Shows only fatal errors."""
)
args = parser.parse_args()

if not os.path.isfile(args.config):
    print(f"FATAL ERROR: Configuration file not found at '{os.path.abspath(args.config)}'.\n")
    parser.print_help()
    exit(1)

if args.lib_path:
    sys.path.insert(0, os.path.abspath(args.lib_path))

try:
    import dynamic_workflows_agents
    from dynamic_workflows_agents import exec_agent, setup_depth_manager
except ImportError:
    print("FATAL ERROR: Could not import the core workflow engine from 'dynamic_workflows_agents.py'.\n")
    parser.print_help()
    exit(1)

# --- UTILITY CLASSES AND FUNCTIONS ---
class CustomJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, bytes):
            try:
                return o.decode('utf-8')
            except UnicodeDecodeError:
                return f"<base64_encoded_bytes>{base64.b64encode(o).decode('utf-8')}</base64_encoded_bytes>"
        return json.JSONEncoder.default(self, o)

def get_nested(data, key_str):
    try:
        return reduce(operator.getitem, key_str.split('.'), data)
    except (KeyError, TypeError, AttributeError):
        return None

# --- HTML REPORT GENERATOR ---
HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Agent Test Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #1c1c1e;
            color: #f2f2f7;
            margin: 0;
            padding: 20px;
        }}
        .container {{
            max-width: 1200px;
            margin: auto;
            background-color: #2c2c2e;
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.4);
        }}
        h1, h2 {{
            border-bottom: 2px solid #3a3a3c;
            padding-bottom: 10px;
        }}
        .summary {{
            display: flex;
            justify-content: space-around;
            padding: 20px;
            background-color: #1c1c1e;
            border-radius: 8px;
            margin-bottom: 20px;
        }}
        .summary-item {{
            text-align: center;
        }}
        .summary-item .value {{
            font-size: 2.5em;
            font-weight: bold;
        }}
        .pass {{ color: #34c759; }}
        .fail {{ color: #ff3b30; }}
        .agent-card {{
            background-color: #3a3a3c;
            border-radius: 8px;
            margin-bottom: 15px;
            overflow: hidden;
        }}
        .agent-header {{
            padding: 15px;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
            background-color: #48484a;
        }}
        .agent-header h3 {{
            margin: 0; font-size: 1.2em;
        }}
        .agent-details {{
            display: none;
            padding: 0 15px 15px 15px;
            border-top: 1px solid #545458;
        }}
        .version-block {{
            margin-top: 15px;
            background-color: #2c2c2e;
            border-radius: 6px;
            border: 1px solid #48484a;
            overflow: hidden;
        }}
        .version-header {{
            padding: 10px 15px;
            background-color: #3a3a3c;
            border-bottom: 1px solid #48484a;
            font-weight: bold;
            cursor: pointer;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}
        .version-details {{
            display: none;
            padding: 0;
        }}
        .test-case {{
            padding: 10px 15px;
            border-bottom: 1px solid #48484a;
        }}
        .test-case:last-child {{
            border-bottom: none;
        }}
        .indicator {{
            display: inline-block;
            width: 70px;
            padding: 5px 0;
            border-radius: 5px;
            text-align: center;
            font-weight: bold;
            font-size: 0.9em;
            margin-right: 15px;
        }}
        .indicator.pass {{ background-color: #34c759; color: #fff; }}
        .indicator.fail {{ background-color: #ff3b30; color: #fff; }}
        code, pre {{
            font-family: "SF Mono", "Menlo", "Monaco", monospace;
            background-color: #1c1c1e;
            border-radius: 5px;
            padding: 2px 5px;
            font-size: 0.9em;
        }}
        pre {{
            padding: 15px;
            white-space: pre-wrap;
            word-wrap: break-word;
        }}
        .error-log {{
            background-color: rgba(255, 59, 48, 0.1);
            border: 1px solid rgba(255, 59, 48, 0.5);
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>&#128270; Agent Test Suite Report</h1>
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 20px;">
            <span>Generated on: {timestamp}</span>
            {service_filter_html}
        </div>
        
        <div class="summary">
            <div class="summary-item">
                <div class="value">{total_agents}</div>
                <div>Agent Families</div>
            </div>
            <div class="summary-item">
                <div class="value">{total_tests}</div>
                <div>Total Assertions</div>
            </div>
            <div class="summary-item pass">
                <div class="value">{passed_tests}</div>
                <div>Passed</div>
            </div>
            <div class="summary-item fail">
                <div class="value">{failed_tests}</div>
                <div>Failed</div>
            </div>
        </div>
        
        <h2>Test Results</h2>
        {results_html}
    </div>

    <script>
        document.querySelectorAll('.agent-header, .version-header').forEach(header => {{
            header.addEventListener('click', (event) => {{
                // Prevent clicks on child elements from toggling the parent
                if (event.target !== header && !header.contains(event.target)) return;

                const details = header.nextElementSibling;
                if (details && (details.classList.contains('agent-details') || details.classList.contains('version-details'))) {{
                    if (details.style.display === 'block') {{
                        details.style.display = 'none';
                    }} else {{
                        details.style.display = 'block';
                    }}
                }}
            }});
        }});
    </script>
</body>
</html>
"""

class TestRunner:
    def __init__(self, config_path='config.json', loglevel='INFO'):
        self.config_path = config_path
        self.loglevel = loglevel
        self.results = {}
        self.stats = {'total_assertions': 0, 'passed_assertions': 0, 'failed_assertions': 0, 'agent_families_with_tests': 0}
        self.config = {}

    def _get_agent_grouping_key(self, full_id: str) -> str:
        match = re.match(r'^(.*)/([^/]+)/v(\d+\.\d+)$', full_id)
        if match:
            return match.group(2)
        return full_id

    def _run_single_test_execution(self, agent_name, agent_data, workflow_inputs):
        temp_config = copy.deepcopy(self.config)
        setup_depth_manager(temp_config)
        dynamic_workflows_agents.log_text_limit = int(
            temp_config.get('workflow_settings', {}).get('log_text_limit', 500)
        )
        
        log_stream = io.StringIO()
        ui_log_handler = logging.StreamHandler(log_stream)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        ui_log_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(getattr(logging, self.loglevel, 'INFO'))
        root_logger.addHandler(ui_log_handler)
        
        final_result_tape, final_status = {}, {"status": {"value": -99, "reason": "Execution did not run"}}
        try:
            final_result_tape, final_status = exec_agent(agent_data, agent_name, config=temp_config, cli_args=workflow_inputs, results={}, force_recompile=True)
        except Exception as e:
            final_result_tape = {"__error__": "An unhandled exception occurred during workflow execution.", "details": str(e)}
            logging.exception("Workflow execution failed")
        finally:
            root_logger.removeHandler(ui_log_handler)
            root_logger.setLevel(original_level)
            
        return final_result_tape, final_status, log_stream.getvalue()

    def _evaluate_assertion(self, assertion, final_result_tape, final_status):
        self.stats['total_assertions'] += 1
        output_variable = assertion["output_variable"]
        assertion_type = assertion["assertion_type"]
        expected_value = assertion["expected_value"]
        
        actual_value = final_status.get('status', {}).get('value') if output_variable == 'status.value' else get_nested(final_result_tape, output_variable)
        
        test_passed = False
        error_msg = ""

        try:
            if actual_value is None:
                error_msg = f"Output variable '{output_variable}' not found in results."
            else:
                if isinstance(actual_value, (dict, list)):
                    actual_str = json.dumps(actual_value, separators=(',', ':'))
                else:
                    actual_str = str(actual_value)
                
                expected_str = str(expected_value)

                if assertion_type == "Equals":
                    normalized_actual = ''.join(actual_str.split())
                    normalized_expected = ''.join(expected_str.split())
                    if normalized_actual == normalized_expected:
                        test_passed = True
                    else:
                        error_msg = f"Expected '{expected_str}', but got '{actual_str}'."
                elif assertion_type == "Regex Match":
                    if re.search(expected_str, actual_str):
                        test_passed = True
                    else:
                        error_msg = f"Regex '{expected_str}' did not match '{actual_str}'."
        except Exception as e:
            error_msg = f"Assertion failed with exception: {e}"

        if test_passed:
            self.stats['passed_assertions'] += 1
        else:
            self.stats['failed_assertions'] += 1
            
        return test_passed, error_msg
    
    def run_all_tests(self):
        print("--- Starting Agent Test Runner ---")
        try:
            with open(self.config_path, 'r') as f:
                self.config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"FATAL: Could not load or parse config file '{self.config_path}': {e}")
            return

        agents = self.config.get('agents', {})
        
        for agent_name, agent_data in agents.items():
            if args.service and args.service not in agent_data.get("web_services", []):
                continue

            run_config = agent_data.get('run_config')
            if not run_config:
                continue

            all_test_cases_for_agent = []
            if "test_cases" in run_config:
                all_test_cases_for_agent.extend(run_config.get("test_cases", []))
            if "tests" in run_config:
                all_test_cases_for_agent.append({
                    "name": "Legacy Test Case",
                    "inputs": run_config.get("last_inputs", {}),
                    "assertions": run_config.get("tests", [])
                })
            
            if not all_test_cases_for_agent or not any(tc.get("assertions") for tc in all_test_cases_for_agent):
                continue

            grouping_key = self._get_agent_grouping_key(agent_name)
            if grouping_key not in self.results:
                self.results[grouping_key] = {'versions': [], 'passed_all': True}

            print(f"Running tests for agent version: '{agent_name}'...")
            
            version_level_results = []
            version_passed_all_cases = True
            
            for test_case in all_test_cases_for_agent:
                inputs = test_case.get("inputs", {})
                assertions = test_case.get("assertions", [])
                if not assertions: continue
                
                final_result, final_status, log = self._run_single_test_execution(agent_name, agent_data, inputs)
                
                assertion_results = []
                case_passed_all_assertions = True
                
                for assertion in assertions:
                    passed, error_msg = self._evaluate_assertion(assertion, final_result, final_status)
                    if not passed:
                        case_passed_all_assertions = False
                    
                    assertion_results.append({
                        'assertion': f"{assertion['output_variable']} [{assertion['assertion_type']}] '{assertion['expected_value']}'",
                        'passed': passed,
                        'error_msg': error_msg
                    })

                if not case_passed_all_assertions:
                    version_passed_all_cases = False

                version_level_results.append({
                    'case_name': test_case.get("name", "Unnamed Test"),
                    'case_passed': case_passed_all_assertions,
                    'assertions': assertion_results,
                    'inputs': inputs,
                    'log': log if not case_passed_all_assertions else "",
                    'final_result': final_result if not case_passed_all_assertions else {}
                })
            
            self.results[grouping_key]['versions'].append({
                'full_name': agent_name,
                'passed_all': version_passed_all_cases,
                'test_cases': version_level_results
            })
            if not version_passed_all_cases:
                self.results[grouping_key]['passed_all'] = False

        self.stats['agent_families_with_tests'] = len(self.results)
        print("--- Test execution complete ---")
        self.generate_report()

    def generate_report(self):
        results_html = ""
        for group_name, group_data in sorted(self.results.items(), key=lambda x: (not x[1]['passed_all'], x[0])):
            group_status_class = "pass" if group_data['passed_all'] else "fail"
            group_status_icon = "&#10004;" if group_data['passed_all'] else "&#10006;"
            
            versions_html = ""
            for version_result in sorted(group_data['versions'], key=lambda x: x['full_name']):
                version_status_class = "pass" if version_result['passed_all'] else "fail"
                version_status_icon = "&#10004;" if version_result['passed_all'] else "&#10006;"
                
                version_cases_html = ""
                for case in version_result['test_cases']:
                    case_status_class = "pass" if case['case_passed'] else "fail"
                    assertions_html = ""
                    for assertion in case['assertions']:
                        indicator_class = "pass" if assertion['passed'] else "fail"
                        indicator_text = "PASS" if assertion['passed'] else "FAIL"
                        error_details = f"<pre><strong>Reason:</strong> {assertion['error_msg']}</pre>" if not assertion['passed'] else ""
                        assertions_html += f"""
                        <div class="assertion-row">
                            <span class="indicator {indicator_class}">{indicator_text}</span>
                            <code>{assertion['assertion']}</code>
                        </div>
                        {error_details}
                        """
                    
                    failure_details_html = ""
                    if not case['case_passed']:
                        failure_details_html = f"""
                        <h4>Inputs Used:</h4>
                        <pre>{json.dumps(case.get('inputs', {}), indent=2)}</pre>
                        <h4>Execution Log ({self.loglevel}):</h4>
                        <pre class="error-log">{case.get('log', '')}</pre>
                        <h4>Final Result JSON:</h4>
                        <pre class="error-log">{json.dumps(case.get('final_result', {}), indent=2, cls=CustomJSONEncoder)}</pre>
                        """

                    version_cases_html += f"""
                    <div class="test-case">
                        <h4 class="{case_status_class}">Test Case: {case.get('case_name', 'Unnamed')}</h4>
                        {assertions_html}
                        {failure_details_html}
                    </div>
                    """
                
                versions_html += f"""
                <div class="version-block">
                    <div class="version-header">
                        <span>{version_result['full_name']}</span>
                        <span class="indicator {version_status_class}">{version_status_icon}</span>
                    </div>
                    <div class="version-details">
                        {version_cases_html}
                    </div>
                </div>
                """

            results_html += f"""
            <div class="agent-card">
                <div class="agent-header">
                    <h3>{group_name}</h3>
                    <span class="indicator {group_status_class}">{group_status_icon}</span>
                </div>
                <div class="agent-details">
                    {versions_html}
                </div>
            </div>
            """
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        final_html = HTML_TEMPLATE.format(
            timestamp=timestamp,
            service_filter_html=f'<p><strong>Service Filter:</strong> <code>{args.service}</code></p>' if args.service else '',
            total_agents=self.stats['agent_families_with_tests'],
            total_tests=self.stats['total_assertions'],
            passed_tests=self.stats['passed_assertions'],
            failed_tests=self.stats['failed_assertions'],
            results_html=results_html
        )
        
        filename = f"Test-Report-{time.strftime('%Y-%m-%d_%H-%M-%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"--- Report Generated: {filename} ---")
        print(f"Summary: {self.stats['passed_assertions']} passed, {self.stats['failed_assertions']} failed out of {self.stats['total_assertions']} total assertions.")

def main():
    runner = TestRunner(config_path=args.config, loglevel=args.loglevel)
    runner.run_all_tests()

if __name__ == "__main__":
    main()