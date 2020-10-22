#!/bin/bash
set -e

sudo apt install -y python3-dev

cd
cd ias_pyrep
pip3 install -r requirements.txt
python3 setup.py install --record files.txt --user

cd ~/ias_ros/src
pip3 install ias_coppelia_sim_core/
pip3 install ias_coppelia_sim_iiwas/

