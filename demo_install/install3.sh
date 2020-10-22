#!/bin/bash
set -e

echo "THIS STEP MUST BE RUN MANUALLY IN A TERMINAL"

source /opt/ros/kinetic/setup.bash
source ~/.bashrc
ias install coppelia_sim_iiwas
ias update
ias make

