import time
import os
import gym
import numpy as np
import XarmEnv
from SAC_with_temperature import Agent
import torch

config = {

    'GUI' : True,
    'reward_type' : "sparse",
}


env = XarmEnv.XarmRobotEnv(config)

lr_actor = 0.001
lr_critic = 0.001
input_dims = 13
n_actions = 3
max_action = 1





agent = Agent(lr_actor,lr_critic,input_dims,n_actions,max_action)

#input dims bi trebao biti 13
#n_actions bi trebao biti 3

#input za actora je 13
#input za critica je 16

#env.make_goal(action)
#env._load_golf_ball()
#time.sleep(.5)
episode_length = 50
num_of_episodes = 10
scores = []
for episode in range(num_of_episodes):
   
        observation = env.reset()
        time_step = 0
        
        
        while time_step < episode_length:
            obs = np.concatenate([observation['observation'],observation['achieved_goal'],observation['desired_goal']],axis=-1) #  ili axis = 0
            obs_tensor = torch.from_numpy(obs).to(agent.actor.device)
            action = agent.choose_action(obs_tensor)
            next_observation, reward, done = env.step(action)

            agent.memory.real_buffer.append(observation,
                                            action,
                                            reward,
                                            done,
                                            next_observation,
                                            1)
            agent.memory.episode_buffer.append(observation,
                                            action,
                                            reward,
                                            done,
                                            next_observation,
                                            1)

            observation = next_observation
            
            if episode > 10:
                real_batch, her_batch = agent.memory.sample()
                agent.learn(real_batch)
                agent.learn(her_batch)

            if done:
                break
            time_step += 1
        score = np.sum(agent.memory.episode_buffer.rewards)
        scores.append(score)

        her_observations = agent.memory.episode_buffer.observations
        her_actions = agent.memory.episode_buffer.actions
        her_next_observations = agent.memory.episode_buffer.next_observations


        end_achieved_episode_goal = observation['achieved_goal']
        # das li ce raditi posto je jedan parametar np.array oblika(time_step, goal_dim) a drugi samo np.array(goal_dim)
        her_rewards = env.compute_reward(her_observations['achieved_goal'],end_achieved_episode_goal)
        her_dones = bool(her_rewards + 1) #ish
        agent.memory.her_buffer.append(her_observations,
                                            her_actions,
                                            her_rewards,
                                            her_dones,
                                            her_next_observations,
                                            time_step)
    
        agent.memory.episode_buffer.reset()
    

        if done:
            break


        
        #print(action)


time.sleep(122)