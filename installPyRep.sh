#!/bin/bash
set -e

### REQUIRES SSH KEY TO ACCESS git.ias.informatik.tu-darmstadt.de REPOSITORIES

cd

sudo apt install -y libffi-dev
sudo apt install -y libopenmpi-dev
sudo apt install -y python3
sudo apt install -y python3-pip
sudo apt install -y wget git
pip3 install --upgrade pip

cur=`pwd`
wget https://coppeliarobotics.com/files/CoppeliaSim_Edu_V4_0_0_Ubuntu16_04.tar.xz
tar -xf CoppeliaSim_Edu_V4_0_0_Ubuntu16_04.tar.xz

echo "export COPPELIASIM_ROOT="$cur/CoppeliaSim_Edu_V4_0_0_Ubuntu16_04"" >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT' >> ~/.bashrc
echo 'export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT' >> ~/.bashrc

export COPPELIASIM_ROOT="$cur/CoppeliaSim_Edu_V4_0_0_Ubuntu16_04"
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$COPPELIASIM_ROOT
export QT_QPA_PLATFORM_PLUGIN_PATH=$COPPELIASIM_ROOT

git clone git@git.ias.informatik.tu-darmstadt.de:keller/ias_pyrep.git
cd ias_pyrep
pip3 install -r requirements.txt --user
python3 setup.py install --user
cd 

pip3 install mpi4py --user
git clone git@git.ias.informatik.tu-darmstadt.de:ias_vrep/scenes/vrep_iiwas.git
cd vrep_iiwas
python3 setup.py install --user

