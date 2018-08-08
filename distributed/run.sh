# Configures the Python 3 virtualenv and runs the desired program
# The flags correspond to:
#     -s: Server Node
#     -w: Worker Node
#     -v: Validator Node

cd ~/Documents/coen_166/neural-networks
source env/bin/activate

if [ "$1" == "-s" ]; then
    python3 dist-train-server.py
elif [ "$1" == "-w" ]; then
    python3 dist-train-worker.py
elif [ "$1" == "-v" ]; then
    python3 dist-train-validator.py
else
    echo "Unknown or no parameter (usage: -s server node; -w worker node)"
fi