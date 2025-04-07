import time

import numpy as np

from SAC_with_temperature_v2 import Agent
import torch
from XarmGolfEnv import XarmRobotGolf
from XarmReach import XarmRobotReach
from tqdm import trange
import time
import wandb

 

#import matplotlib.pyplot as plt
config = {

    'GUI' : False,
    'reward_type' : "sparse",
}
test_config = {
     'GUI' : True,
     'reward_type' : "sparse", 
}

wandb.init(

    project = "Xarm-golf",
    config = {
        "lr_actor": 0.0001,
        "lr_critic": 0.003,
        "batch_size": 2048,
        "nn_dims": 512,
        "temperature": 0.3,
        "episodes": 30000,
        "entropy": -2,
        "init weights": "Xavier_uniform_",
        "optimizer": "AdamW",
        "gamma": 0.95
    },
    id= 'test36 - camera  test 3 - pos + vel'
)  
env =XarmRobotGolf(config)
#test_env = XarmRobotGolf(test_config)
lr_actor = 0.0001
lr_critic = 0.003
input_dims = 18
obs_dims = 12
n_actions = 4
max_action = 1

agent = Agent(lr_actor,lr_critic,input_dims,obs_dims,n_actions,max_action,fc1_dim=512,fc2_dim=512,batch_size=2048,gamma=0.95)


test_scores = []
test_episode_count = 0

@torch.no_grad
def test():
    with torch.no_grad():
        test_env = XarmRobotGolf(test_config)
        test_env.phase = env.phase
        #test_env.test_reset()
        agent.evaluate_mode()
        test_episode_range = 100
        made = 0
        for test_episode in trange(test_episode_range):
            test_observation = test_env.reset()
            test_time_step = 0
            test_score = 0
            while test_time_step < 50:
                test_obs = np.concatenate([test_observation['observation'],test_observation['achieved_goal'],test_observation['desired_goal']],axis=-1,dtype=np.float32) #  ili axis = 0
                test_action = agent.choose_action(test_obs,True)
                test_next_observation, test_reward, test_done, _ = test_env.step(test_action)
                test_observation = test_next_observation
                
                test_score +=  test_reward
                if test_done:
                    break
                test_time_step += 1
                time.sleep(1./30)
            test_scores.append(test_score)
            if test_score> -50:
                made+=1
            print(f"Test Episode {len(test_scores)}, score - {test_score}")
        agent.training_mode()
        test_env.close()
        print(f"Test 100 average score: {np.average(test_scores[-100:])}")
        print(f"Success rate - {made}%")


episode_length = 50
num_of_episodes = 30000
scores = []
actor_losses = []
critic_losses = []
temperature_losses = []
loss = []

als = []
cls = []
tls = []




for episode in trange(num_of_episodes):
        observation = env.reset()
        time_step = 0
        episode_score = 0
        agent.last_position = None
        agent.last_time = None

        
        while time_step < episode_length:
            #obs = np.concatenate([observation['observation'],observation['achieved_goal'],observation['desired_goal']],axis=-1,dtype=np.float32) #  ili axis = 0
            action = agent.choose_action(observation, time_step)
            next_observation, reward, done, info = env.step(action)
            
            agent.real_memory_append(observation,
                                     action,
                                     reward,
                                     done,
                                     next_observation,
                                     time_step,
                                     1)
           
            if episode > 50:
                batch = agent.memory.sample()   # napravi funkciju za ovo
                al, cl, tl ,alpha,log_alpha= agent.learn(batch)
                #wandb.log({"actor_loss": al, "critic_loss": cl, "temp_loss": tl})
            episode_score += reward
            time_step += 1#pogledaj da li ovo ide pre ili posle breaka
            if done:
                break
            
            observation = next_observation
        
        
        scores.append(episode_score)
        avg_score = np.mean(scores[-100:])
        print(f"Episode:, {episode}, score: {episode_score}, average score: {avg_score}")
        
        if episode > 50:
            wandb.log({"score": avg_score,"actor_loss": al, "critic_loss": cl, "temp_loss": tl, "alpha": alpha, "log_alpha":log_alpha})
        
        #if (episode)% 1000 == 0:
        #    test()
        
        #end_achieved_episode_goal = observation['achieved_goal']


        her_achieved_goals= agent.memory.real_buffer.return_achieved_goals(time_step)   # napravi funkciju za ovo
        her_rewards, her_dones = env.compute_her_reward(her_achieved_goals[-1], her_achieved_goals)
        agent.her_memory_append(time_step,her_rewards,her_dones)


        """
        for index in range(time_step):


            her_obs = {
                'observation': observations[index],
                'achieved_goal': achieved_goals[index],
                'desired_goal': end_achieved_episode_goal
            }
            her_next_obs = {
                'observation': next_observations[index],
                'achieved_goal': next_achieved_goals[index],
                'desired_goal': end_achieved_episode_goal
            }
            
            her_reward = env.compute_reward(achieved_goals[index],end_achieved_episode_goal)
            her_done = her_reward + 1 #ish
            agent.memory.her_buffer.append(her_obs,
                                                actions[index],
                                                her_reward,
                                                her_done,
                                                her_next_obs,
                                                1)

        """

        if episode == 3000:
            env.phase = 2    
        if episode == 10000:
            env.phase = 3
        """
        observations = []
        achieved_goals=[]
        desired_goals =[]
        actions = []
        rewards = []
        next_observations = []
        next_achieved_goals=[]
        next_desired_goals =[]
        """
        

wandb.finish()
