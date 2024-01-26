#!/bin/bash
apt update && upgrade -y
apt install python3-venv libopenblas-dev -y
mkdir cubertEnv
python3 -m venv cubertEnv
