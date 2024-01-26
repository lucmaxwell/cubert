#!/bin/bash
sudo apt update && upgrade -y
sudo apt install python3-venv libopenblas-dev -y
mkdir cubertEnv
python3 -m venv cubertEnv
source ./cubertEnv/bin/activate
pip install -r requirements.txt
pip list
