#!/bin/bash
set -e

# requires pyrep, pip, git

sudo apt install -y python3-tk
sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev

if [ -z ${COPPELIASIM_ROOT+x} ]; then 
printf "COPPELIASIM_ROOT is not set\\nrun PyRep installation or\\nsource .bashrc or\\use a new terminal first\\n"; 
else 
echo "COPPELIASIM_ROOT is set to '$COPPELIASIM_ROOT'"; 
fi

echo 'export QT_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc
export QT_PLUGIN_PATH=$COPPELIASIM_ROOT


# Installing real_robots
git clone https://github.com/emilio-cartoni/real_robots.git
cd real_robots
git checkout PyRep
pip3 install -e . --user
cd ..

# Install pyrepgym
git clone https://github.com/francesco-mannella/pyrepgym.git
cd pyrepgym
git checkout ROSVersion
pip3 install -e . --user
cd ..

git clone https://github.com/emilio-cartoni/REAL2020_starter_kit.git
cd REAL2020_starter_kit
git checkout PyRep
cd ..
pip3 install PyYaml --user
pip3 install tensorflow --user
pip3 install opencv-python --user
pip3 install matplotlib --user




