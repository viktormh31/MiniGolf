import time

import numpy as np

from SAC_with_temperature_v2 import Agent
import torch
from XarmGolfEnv import XarmRobotGolf
from tqdm import trange
import time



test_config = {
     'GUI' : True,
     'reward_type' : "sparse", 
}

#test_env = XarmRobotGolf(test_config)
lr_actor = 0.0001
lr_critic = 0.003
input_dims = 18
obs_dims = 12
n_actions = 4
max_action = 1

agent = Agent(lr_actor,lr_critic,input_dims,obs_dims,n_actions,max_action,fc1_dim=512,fc2_dim=512,batch_size=2048,gamma=0.95)

agent.load_models(90000)


test_scores = []
test_episode_count = 0

#@torch.no_grad
def test(phase):
    with torch.no_grad():
        test_env = XarmRobotGolf(test_config)
        test_env.phase = phase
        test_env.reset()
        agent.evaluate_mode()
        test_episode_range = 200
        made = 0.
        for test_episode in trange(test_episode_range):
            test_observation = test_env.reset()
            test_time_step = 0
            test_score = 0
            agent.last_position = None
            agent.last_time = None

            while test_time_step < 50:
                action = agent.choose_action(test_observation, test_time_step,True)
                test_next_observation, test_reward, test_done, _ = test_env.step(action)
            

                
                test_score +=  test_reward
                test_time_step += 1
                if test_done:
                    break
                
                test_observation = test_next_observation
                #time.sleep(1./30)
            test_scores.append(test_score)
            if test_score> -50:
                made+=1.
            print(f"Test Episode {len(test_scores)}, score - {test_score}")
        agent.training_mode()
        test_env.close()
        print(f"Test 200 average score: {np.average(test_scores[-200:])}")
        print(f"Success rate - {made/2.}%")
        
test(3)
