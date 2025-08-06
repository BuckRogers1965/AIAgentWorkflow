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

# --- CORE LIBRARY IMPORTS ---
try:
    import dynamic_workflows_agents
    from dynamic_workflows_agents import exec_agent, setup_depth_manager
except ImportError:
    print("FATAL ERROR: Could not import the core workflow engine from 'dynamic_workflows_agents.py'.")
    print("Please ensure this script is in the same directory as the core library.")
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
        <p>Generated on: {timestamp}</p>
        
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
            run_config = agent_data.get('run_config')
            if not run_config or not run_config.get('tests'):
                continue

            self.stats['agents_with_tests'] += 1
            print(f"Running tests for agent: '{agent_name}'...")
            self.run_single_agent_test(agent_name, agent_data, config)
        
        print("--- Test execution complete ---")
        self.generate_report()

    def run_single_agent_test(self, agent_name, agent_data, config):
        run_config = agent_data['run_config']
        tests = run_config['tests']
        
        workflow_inputs = run_config.get('last_inputs', {})
        
        temp_config = copy.deepcopy(config)
        setup_depth_manager(temp_config)
        dynamic_workflows_agents.log_text_limit = int(
            temp_config.get('workflow_settings', {}).get('log_text_limit', 500)
        )
        
        #agent_to_run = None
        #if agent_data.get('type') != 'workflow':
            #agent_to_run = create_temp_workflow(agent_name, agent_data, temp_config, workflow_inputs)
        #else:
            #agent_to_run = agent_data

        log_stream = io.StringIO()
        ui_log_handler = logging.StreamHandler(log_stream)
        formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
        ui_log_handler.setFormatter(formatter)
        root_logger = logging.getLogger()
        original_level = root_logger.level
        root_logger.setLevel(getattr(logging, self.loglevel, logging.INFO))
        root_logger.addHandler(ui_log_handler)
        
        final_result_tape, final_status = {}, {"status": {"value": -99, "reason": "Execution did not run"}}
        #print (agent_data, agent_name)
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
            
            # --- START OF THE REAL, SIMPLER FIX ---
            if output_variable == 'status.value':
                actual_value = actual_value = final_status.get('status', {}).get('value')
            else:
                actual_value = get_nested(final_result_tape, output_variable)
            # --- END OF THE REAL, SIMPLER FIX ---

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
        # ... (HTML generation is unchanged) ...
        results_html = ""
        for result in sorted(self.results, key=lambda x: (not x['passed_all'], x['agent_name'])):
            status_class = "pass" if result['passed_all'] else "fail"
            status_icon = "&#10004;" if result['passed_all'] else "&#10006;"
            
            details_html = ""
            for test in result['tests']:
                indicator_class = "pass" if test['passed'] else "fail"
                indicator_text = "PASS" if test['passed'] else "FAIL"
                
                error_details = ""
                if not test['passed']:
                    error_details = f"<pre><strong>Reason:</strong> {test['error_msg']}</pre>"

                details_html += f"""
                <div class="test-case">
                    <span class="indicator {indicator_class}">{indicator_text}</span>
                    <code>{test['assertion']}</code>
                    {error_details}
                </div>
                """
            
            if not result['passed_all']:
                details_html += f"""
                <h4>Execution Log ({self.loglevel}):</h4>
                <pre class="error-log">{result['log']}</pre>
                <h4>Final Result JSON:</h4>
                <pre class="error-log">{json.dumps(result['final_result'], indent=2, cls=CustomJSONEncoder)}</pre>
                """

            results_html += f"""
            <div class="agent-card">
                <div class="agent-header">
                    <h3>{result['agent_name']}</h3>
                    <span class="indicator {status_class}">{status_icon}</span>
                </div>
                <div class="agent-details">
                    {details_html}
                </div>
            </div>
            """
        
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        final_html = HTML_TEMPLATE.format(
            timestamp=timestamp,
            total_agents=self.stats['agents_with_tests'],
            total_tests=self.stats['total'],
            passed_tests=self.stats['passed'],
            failed_tests=self.stats['failed'],
            results_html=results_html
        )
        
        filename = f"Test-Report-{time.strftime('%Y-%m-%d_%H-%M-%S')}.html"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(final_html)
            
        print(f"--- Report Generated: {filename} ---")
        print(f"Summary: {self.stats['passed']} passed, {self.stats['failed']} failed out of {self.stats['total']} tests.")

def main():
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
    
    runner = TestRunner(config_path=args.config, loglevel=args.loglevel)
    runner.run_all_tests()

if __name__ == "__main__":
    main()
