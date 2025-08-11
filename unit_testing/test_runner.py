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
# This is the ONLY section that is different from your original file.

# 1. DEFINE THE PARSER
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

# 2. VALIDATE CONFIG FILE PATH
if not os.path.isfile(args.config):
    print(f"FATAL ERROR: Configuration file not found at '{os.path.abspath(args.config)}'.\n")
    parser.print_help()
    exit(1)

# 3. PREPARE LIBRARY PATH
if args.lib_path:
    sys.path.insert(0, os.path.abspath(args.lib_path))

# 4. IMPORT CORE LIBRARY WITH CORRECT ERROR HANDLING
try:
    import dynamic_workflows_agents
    from dynamic_workflows_agents import exec_agent, setup_depth_manager
except ImportError:
    print("FATAL ERROR: Could not import the core workflow engine from 'dynamic_workflows_agents.py'.\n")
    parser.print_help()
    exit(1)


# --- UTILITY CLASSES AND FUNCTIONS (UNCHANGED) ---
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

# --- HTML REPORT GENERATOR (UNCHANGED) ---
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
            transition: all 0.2s ease-in-out;
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
            margin: 0;
        }}
        .agent-details {{
            display: none;
            padding: 15px;
            border-top: 1px solid #545458;
        }}
        .test-case {{
            padding: 10px;
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
                <div>Agents Found</div>
            </div>
            <div class="summary-item">
                <div class="value">{total_tests}</div>
                <div>Tests Executed</div>
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
        document.querySelectorAll('.agent-header').forEach(header => {{
            header.addEventListener('click', () => {{
                const details = header.nextElementSibling;
                if (details.style.display === 'block') {{
                    details.style.display = 'none';
                }} else {{
                    details.style.display = 'block';
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
        self.results = []
        self.stats = {'total': 0, 'passed': 0, 'failed': 0, 'agents_with_tests': 0}

    def run_all_tests(self):
        print("--- Starting Agent Test Runner ---")
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"FATAL: Could not load or parse config file '{self.config_path}': {e}")
            return

        agents = config.get('agents', {})
        
        for agent_name, agent_data in agents.items():
            if args.service:
                agent_services = agent_data.get("web_services", [])
                if args.service not in agent_services:
                    continue

            run_config = agent_data.get('run_config')
            if not run_config:
                continue
    
            if "test_cases" in run_config: # This is the NEW multi-test format.
                self.stats['agents_with_tests'] += 1
                print(f"Running new tests for agent: '{agent_name}'...")
                self.run_multi_case_tests_for_agent(agent_name, agent_data, config)
            elif "tests" in run_config: # This is the OLD single-test format.
                if not run_config.get('tests'):
                    continue
                self.stats['agents_with_tests'] += 1
                print(f"Running tests for agent: '{agent_name}'...")
                self.run_single_agent_test(agent_name, agent_data, config)
        
        print("--- Test execution complete ---")
        self.generate_report()

    def run_multi_case_tests_for_agent(self, agent_name, agent_data, config):
        run_config = agent_data['run_config']
        test_cases = run_config.get("test_cases", [])
        
        agent_level_results = []
        agent_passed_all_cases = True
        
        for test_case in test_cases:
            # Prepare a temporary run_config for the single-test runner
            temp_run_config = {
                "last_inputs": test_case.get("inputs", {}),
                "tests": test_case.get("assertions", [])
            }
            temp_agent_data = copy.deepcopy(agent_data)
            temp_agent_data['run_config'] = temp_run_config

            # This is the original, unchanged single-test logic
            #----------------------------------------------------------------
            tests = temp_run_config['tests']
            workflow_inputs = temp_run_config.get('last_inputs', {})
            
            temp_config = copy.deepcopy(config)
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
            root_logger.setLevel(getattr(self, 'loglevel', 'INFO'))
            root_logger.addHandler(ui_log_handler)
            
            final_result_tape, final_status = {}, {"status": {"value": -99, "reason": "Execution did not run"}}
            try:
                final_result_tape, final_status = exec_agent(temp_agent_data, agent_name, config=temp_config, cli_args=workflow_inputs, results={})
            except Exception as e:
                final_result_tape = {"__error__": "An unhandled exception occurred during workflow execution.", "details": str(e)}
                logging.exception("Workflow execution failed")
            finally:
                root_logger.removeHandler(ui_log_handler)
                root_logger.setLevel(original_level)
            #----------------------------------------------------------------

            # Process results for this specific test case
            assertion_results = []
            case_passed_all_assertions = True
            
            for test in tests:
                self.stats['total'] += 1
                
                output_variable = test["output_variable"]
                actual_value = final_status.get('status', {}).get('value') if output_variable == 'status.value' else get_nested(final_result_tape, output_variable)
                expected_value = test["expected_value"]
                assertion_type = test["assertion_type"]
                
                test_passed = False
                error_msg = ""
                try:
                    if assertion_type == "Regex Match":
                        if actual_value is not None and re.search(str(expected_value), str(actual_value)):
                            test_passed = True
                        else:
                            error_msg = f"Regex '{expected_value}' did not match '{actual_value}'."
                    elif assertion_type == "Equals":
                        if actual_value is not None and str(actual_value) == str(expected_value):
                            test_passed = True
                        else:
                            error_msg = f"Expected string '{expected_value}', but got string '{actual_value}'."
                except Exception as e:
                    error_msg = f"Assertion failed with exception: {e}"

                if test_passed:
                    self.stats['passed'] += 1
                else:
                    self.stats['failed'] += 1
                    case_passed_all_assertions = False
                
                assertion_results.append({
                    'assertion': f"{output_variable} [{assertion_type}] '{expected_value}'",
                    'passed': test_passed,
                    'error_msg': error_msg
                })

            if not case_passed_all_assertions:
                agent_passed_all_cases = False

            agent_level_results.append({
                'case_name': test_case.get("name", "Unnamed Test"),
                'case_passed': case_passed_all_assertions,
                'assertions': assertion_results,
                'inputs': workflow_inputs,
                'log': log_stream.getvalue() if not case_passed_all_assertions else "",
                'final_result': final_result_tape if not case_passed_all_assertions else {}
            })

        # Append the aggregated results for the entire agent
        self.results.append({
            'agent_name': agent_name,
            'passed_all': agent_passed_all_cases,
            'test_cases': agent_level_results
        })

    def run_single_agent_test(self, agent_name, agent_data, config):
        run_config = agent_data['run_config']
        tests = run_config['tests']
        
        workflow_inputs = run_config.get('last_inputs', {})
        
        temp_config = copy.deepcopy(config)
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
        root_logger.setLevel(getattr(logging, self.loglevel, logging.INFO))
        root_logger.addHandler(ui_log_handler)
        
        final_result_tape, final_status = {}, {"status": {"value": -99, "reason": "Execution did not run"}}
        try:
            final_result_tape, final_status = exec_agent(agent_data, agent_name, config=temp_config, cli_args=workflow_inputs, results={})
        except Exception as e:
            final_result_tape = {"__error__": "An unhandled exception occurred during workflow execution.", "details": str(e)}
            logging.exception("Workflow execution failed")
        finally:
            root_logger.removeHandler(ui_log_handler)
            root_logger.setLevel(original_level)
            
        agent_test_results = []
        agent_passed_all = True
        
        for test in tests:
            self.stats['total'] += 1
            
            output_variable = test["output_variable"]
            actual_value = None
            
            if output_variable == 'status.value':
                actual_value = final_status.get('status', {}).get('value')
            else:
                actual_value = get_nested(final_result_tape, output_variable)

            expected_value = test["expected_value"]
            assertion_type = test["assertion_type"]
            
            test_passed = False
            error_msg = ""
            try:
                if assertion_type == "Regex Match":
                    if actual_value is not None and re.search(str(expected_value), str(actual_value)):
                        test_passed = True
                    else:
                        error_msg = f"Regex '{expected_value}' did not match '{actual_value}'."
                elif assertion_type == "Equals":
                    if actual_value is not None and str(actual_value) == str(expected_value):
                        test_passed = True
                    else:
                        error_msg = f"Expected string '{expected_value}', but got string '{actual_value}'."
            except Exception as e:
                error_msg = f"Assertion failed with exception: {e}"

            if test_passed:
                self.stats['passed'] += 1
            else:
                self.stats['failed'] += 1
                agent_passed_all = False
            
            agent_test_results.append({
                'assertion': f"{test['output_variable']} [{assertion_type}] '{expected_value}'",
                'passed': test_passed,
                'actual_value': actual_value,
                'error_msg': error_msg
            })

        self.results.append({
            'agent_name': agent_name,
            'passed_all': agent_passed_all,
            'tests': agent_test_results,
            'log': log_stream.getvalue() if not agent_passed_all else "",
            'final_result': final_result_tape if not agent_passed_all else {}
        })

    def generate_report(self):
        results_html = ""
        for agent_result in sorted(self.results, key=lambda x: (not x.get('passed_all', True), x['agent_name'])):
            status_class = "pass" if agent_result.get('passed_all', True) else "fail"
            status_icon = "&#10004;" if agent_result.get('passed_all', True) else "&#10006;"
            
            # This is the new nested structure for the report
            test_cases_html = ""
            if 'test_cases' in agent_result: # Handle new format
                for case in agent_result['test_cases']:
                    case_status_class = "pass" if case.get('case_passed', True) else "fail"
                    
                    assertions_html = ""
                    for assertion in case.get('assertions', []):
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
                    if not case.get('case_passed', True):
                        failure_details_html = f"""
                        <h4>Inputs Used:</h4>
                        <pre>{json.dumps(case.get('inputs', {}), indent=2)}</pre>
                        <h4>Execution Log ({self.loglevel}):</h4>
                        <pre class="error-log">{case.get('log', '')}</pre>
                        <h4>Final Result JSON:</h4>
                        <pre class="error-log">{json.dumps(case.get('final_result', {}), indent=2, cls=CustomJSONEncoder)}</pre>
                        """

                    test_cases_html += f"""
                    <div class="test-case">
                        <h4 class="{case_status_class}">Test Case: {case.get('case_name', 'Unnamed')}</h4>
                        {assertions_html}
                        {failure_details_html}
                    </div>
                    """
            elif 'tests' in agent_result: # Handle old format for backward compatibility
                for test in agent_result['tests']:
                    indicator_class = "pass" if test['passed'] else "fail"
                    indicator_text = "PASS" if test['passed'] else "FAIL"
                    error_details = f"<pre><strong>Reason:</strong> {test['error_msg']}</pre>" if not test['passed'] else ""
                    test_cases_html += f"""
                    <div class="test-case">
                        <span class="indicator {indicator_class}">{indicator_text}</span>
                        <code>{test['assertion']}</code>
                        {error_details}
                    </div>
                    """
                if not agent_result.get('passed_all', True):
                    test_cases_html += f"""
                    <h4>Execution Log ({self.loglevel}):</h4>
                    <pre class="error-log">{agent_result.get('log', '')}</pre>
                    <h4>Final Result JSON:</h4>
                    <pre class="error-log">{json.dumps(agent_result.get('final_result', {}), indent=2, cls=CustomJSONEncoder)}</pre>
                    """

            results_html += f"""
            <div class="agent-card">
                <div class="agent-header">
                    <h3>{agent_result['agent_name']}</h3>
                    <span class="indicator {status_class}">{status_icon}</span>
                </div>
                <div class="agent-details">
                    {test_cases_html}
                </div>
            </div>
            """
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        final_html = HTML_TEMPLATE.format(
            timestamp=timestamp,
            service_filter_html=f'<p><strong>Service Filter:</strong> <code>{args.service}</code></p>' if args.service else '',
            total_agents=self.stats.get('agents_with_tests', self.stats.get('agents_tested', 0)),
            total_tests=self.stats.get('total', self.stats.get('test_cases', 0)),
            passed_tests=self.stats.get('passed', self.stats.get('assertions_passed', 0)),
            failed_tests=self.stats.get('failed', self.stats.get('assertions_failed', 0)),
            results_html=results_html
        )
        
        filename = f"Test-Report-{time.strftime('%Y-%m-%d_%H-%M-%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"--- Report Generated: {filename} ---")
        passed_count = self.stats.get('passed', self.stats.get('assertions_passed', 0))
        failed_count = self.stats.get('failed', self.stats.get('assertions_failed', 0))
        total_count = passed_count + failed_count
        print(f"Summary: {passed_count} passed, {failed_count} failed out of {total_count} total assertions.")


def main():
    runner = TestRunner(config_path=args.config, loglevel=args.loglevel)
    runner.run_all_tests()

if __name__ == "__main__":
    main()
