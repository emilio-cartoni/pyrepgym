#!/bin/bash
sed -i 's/LEFT/RIGHT/g' pyrepgym/envs/PyRepEnv.py
sed -i 's/left/right/g' pyrepgym/envs/PyRepEnv.py
sed -i 's/move_duration=np.array(\[3\]/move_duration=np.array(\[10\]/g' pyrepgym/envs/PyRepEnv.py
