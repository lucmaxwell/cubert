#!/bin/bash
pip3 install -U pip setuptools wheel
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip3 install -r requirements.txt
pip3 list
