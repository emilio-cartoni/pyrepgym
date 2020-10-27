# pyrepgym

**INSTALLATION**

Download all install&lt;something&gt;.sh files from the demo_install directory.  
Put them in your home directory (other locations not tested).  
Delete the pyrepgym repository if you downloaded the whole repository, this will be automatically downloaded and installed later.  

Execute:  
./installPyRep.sh  
./installROS.sh  
./installREAL.sh  

You may need to open a new terminal between running each install file.  
You may skip installPyRep.sh and installROS.sh if you have already PyRep+ROS configured.  


**RUNNING A MOVEMENT DEMO ON SIMULATION**

It is possible to run a movement demo by running:  
- roscore  
- simlaunch.py  
- demo.py  

Demo.py will make 5 random movements over the table.  

**RUNNING THE BASELINE IN SIMULATION**

Launch:  
- roscore  
- simlaunch.py  
- local\_evaluation.py (in REAL2020_starter_kit)  

You can edit local\_evaluation.py to run either the intrinsic phase or the extrinsic phase, or to modify their length.  
For the intrinsic phase, just change the number of steps (each step is 1 action).  
For the extrinsic phase, you can set both the number of trials (number of goals to pursue) and the length of each trial (how many actions are allowed for each goal).  

**RUNNING THE BASELINE ON THE PHYSICAL ROBOT (UNTESTED)**
Launch:  
- roscore  
- imageGenerator.py _not available yet!_  
- local\_evaluation.py (in REAL2020_starter_kit)  

**TESTING WITH IMAGE GENERATOR IN SIM (TBD)**
Launch:  
- roscore  
- simlaunch_testimage.py _not available yet!_  
- imageGenerator.py _not available yet!_  
- local\_evaluation.py (in REAL2020_starter_kit)  








