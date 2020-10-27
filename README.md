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
- demos/demo.py  

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
- simlaunch.py (set publishImage = False)  
- imageGenerator.py_  
- local\_evaluation.py (in REAL2020_starter_kit)  


**NOTES:**
1) The current movement should be a linear trajectory between two points on the table;  
however, since the robot is actually instructed to go from one joint position (corresponding to the first point) to another joint position (corresponding to second point) it is not garaunteed that the trajectory is linear. **This can also cause the robot to hit the shelf if the action_space limits are close to it!**  

2) PyRepEnv.py, simlaunch.py and imageGenerator.py have IMAGE\_TOPIC\_NAME and OBJPOS\_TOPIC\_NAME variables to change ROS topic names for the image and object position messagges.








