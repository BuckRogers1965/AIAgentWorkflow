import requests
import time
import xml.etree.ElementTree as ET
import statistics
import logging

# --- Configuration ---
URL = "http://localhost:5000/execute/append_text?whole_text=1111&part_text=222222"
NUM_REQUESTS = 100000
REQUEST_TIMEOUT = 10 # seconds

def run_benchmark():
    """
    Sends a specified number of requests to the agent web service and reports on performance.
    """
    print(f"--- Starting Benchmark ---")
    print(f"Target URL: {URL}")
    print(f"Number of Requests: {NUM_REQUESTS}\n")

    timings = []
    success_count = 0
    error_count = 0
    last_successful_response = None
    
    # --- Main request loop ---
    for i in range(NUM_REQUESTS):
        start_time = time.perf_counter()
        response = None # Initialize response to None before the try block
        status_value = None # Initialize status_value to None
        
        try:
            # Send the GET request
            response = requests.get(URL, timeout=REQUEST_TIMEOUT)
            
            # Check for HTTP success first
            response.raise_for_status() # This will raise an HTTPError for bad responses (4xx or 5xx)
            
            # If HTTP was successful, check for application-level success by parsing the XML
            root = ET.fromstring(response.content)
            status_value = root.findtext('./status/value')
            
            if status_value == '0':
                success_count += 1
                last_successful_response = response.text
            else:
                error_count += 1
                reason = root.findtext('./status/reason', 'Unknown application error')
                print(f"ERROR on request {i+1}: Application error - Status {status_value}, Reason: {reason}")

        except requests.exceptions.RequestException as e:
            error_count += 1
            print(f"ERROR on request {i+1}: Request failed - {e}")
        
        end_time = time.perf_counter()
        
        # Only record timings if both the request and the application logic succeeded
        if response and response.ok and status_value == '0':
            timings.append(end_time - start_time)

        # Print a progress indicator
        if (i + 1) % 100 == 0:
            print(f"  ...completed {i + 1}/{NUM_REQUESTS} requests.")

    print("\n--- Benchmark Complete ---")

    # --- Print summary ---
    if not timings:
        print("No successful requests were made. Cannot calculate statistics.")
    else:
        # Convert seconds to milliseconds for readability
        timings_ms = [t * 1000 for t in timings]
        
        best_time = min(timings_ms)
        worst_time = max(timings_ms)
        mean_time = statistics.mean(timings_ms)

        print("\n================== RESULTS ==================")
        print(f"Total Requests Sent: {NUM_REQUESTS}")
        print(f"Successful Requests: {success_count}")
        print(f"Failed Requests:     {error_count}")
        print("-------------------------------------------")
        print(f"Fastest Request: {best_time:.4f} ms")
        print(f"Slowest Request: {worst_time:.4f} ms")
        print(f"Mean Request Time: {mean_time:.4f} ms")
        print("===========================================")

    # --- Print last successful response ---
    if last_successful_response:
        print("\n--- Last Successful Response ---")
        print(last_successful_response)
    else:
        print("\n--- No successful responses were received to display. ---")


if __name__ == "__main__":
    try:
        import requests
    except ImportError:
        print("Error: The 'requests' library is not installed.")
        print("Please install it by running: pip install requests")
        exit(1)

    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
        
    run_benchmark()
