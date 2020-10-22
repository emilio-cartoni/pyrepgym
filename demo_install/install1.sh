#!/bin/bash
set -e

### REQUIRES SSH KEY TO ACCESS git.ias.informatik.tu-darmstadt.de REPOSITORIES

sudo apt-get install -y curl wget git
cd
curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
python3 get-pip.py --user
python2 get-pip.py --user


cd
sudo apt install -y python-dev libffi-dev libavcodec-dev libavformat-dev libswscale-dev
git clone git@git.ias.informatik.tu-darmstadt.de:ias_coppelia_sim/ias_pyrep.git
cd ias_pyrep
cur=`pwd`
wget https://coppeliarobotics.com/files/CoppeliaSim_Edu_V4_0_0_Ubuntu16_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_0_0_Ubuntu16_04.tar.xz
echo "export COPPELIASIM_ROOT="$cur/CoppeliaSim_Edu_V4_0_0_Ubuntu16_04"" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT:$COPPELIASIM_ROOT/platforms' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc

export COPPELIASIM_ROOT="$cur/CoppeliaSim_Edu_V4_0_0_Ubuntu16_04"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

pip2 install -r requirements.txt
python2 setup.py install --record files.txt --user  

