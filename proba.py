import time
import os
import gym
import numpy as np
import XarmEnv

config = {

    'GUI' : True,
    'reward_type' : "sparse",
}


env = XarmEnv.XarmRobotEnv(config)
action = np.array([0.5,0.0,0.3])
#env.make_goal(action)
#env._load_golf_ball()
#time.sleep(.5)
env._get_obs()
for i in range(50000):
    env.move_joint(10)
    
    time.sleep(.01)
    #print(action)
time.sleep(122)