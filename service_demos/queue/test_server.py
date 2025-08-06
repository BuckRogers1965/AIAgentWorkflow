# test_server.py
# A minimal Flask server to test the environment.

try:
    from flask import Flask

    app = Flask(__name__)

    @app.route('/')
    def hello_world():
        # This just returns a simple, static HTML string.
        # There are no variables or complex formatting.
        return """
        <html>
            <head><title>Test Server</title></head>
            <body>
                <h1>Success!</h1>
                <p>If you can see this page, your Flask installation and Python environment are working correctly.</p>
                <p>The problem is likely within the queue_server.py application code.</p>
            </body>
        </html>
        """

    if __name__ == '__main__':
        print("[+] Starting minimal test server...")
        print("[+] Please open http://127.0.0.1:5001 in your browser.")
        # Running on port 5001 as requested.
        app.run(host='127.0.0.1', port=5001)

except ImportError as e:
    print("--- IMPORT ERROR ---")
    print(f"Failed to import a critical library: {e}")
    print("This suggests a problem with the Python environment or installation.")

except Exception as e:
    print("--- UNEXPECTED STARTUP ERROR ---")
    print(f"An error occurred before the server could even start: {e}")