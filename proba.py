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
obss = []
obss_ = []
rewards = []
dones = []



for episode in range(10):
    obs = env.reset()
    for i in range(50):
        obs = env._get_obs()
        env.move_joint(10)
        #action = env.action_space.sample()
        obs_, reward,done = env.step(action)
        
        
        obss.append(obs)
        obss_.append(obs_)
        rewards.append(reward)
        dones.append(done)

        if done:
            break


        time.sleep(2/120)
        #print(action)
time.sleep(122)