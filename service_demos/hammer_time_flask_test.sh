#!/bin/bash

echo "Starting 10 concurrent log/clients in the background..."

python3 flask_web_test.py > log/client_1.log &
python3 flask_web_test.py > log/client_2.log &
python3 flask_web_test.py > log/client_3.log &
python3 flask_web_test.py > log/client_4.log &
python3 flask_web_test.py > log/client_5.log &
python3 flask_web_test.py > log/client_6.log &
python3 flask_web_test.py > log/client_7.log &
python3 flask_web_test.py > log/client_8.log &
python3 flask_web_test.py > log/client_9.log &
python3 flask_web_test.py > log/client_10.log &

echo "All log/clients launched. Use 'jobs' to see their status."
echo "Use 'wait' to pause until they are all complete."
echo "Output is being redirected to log/client_*.log files."
