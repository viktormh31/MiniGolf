import time
import numpy as np

from SAC_with_temperature_v2 import Agent
import torch
from XarmGolfEnv import XarmRobotGolf
from XarmReach import XarmRobotReach
from tqdm import trange
import time
import sys
import os




#import matplotlib.pyplot as plt
config = {

    'GUI' : False,
    'reward_type' : "sparse",
}
test_config = {
     'GUI' : True,
     'reward_type' : "sparse", 
}


env =XarmRobotGolf(config)
#test_env = XarmRobotGolf(test_config)
lr_actor = 0.0005
lr_critic = 0.0005
input_dims = 27
obs_dims = 21
n_actions = 4
max_action = 1

test_scores = []
test_episode_count = 0

def test():
    with torch.no_grad():
        test_env = XarmRobotGolf(test_config)
        test_env.phase = 2
        #test_env.test_reset()
        agent.evaluate_mode()
        test_episode_range = 3 
        for test_episode in range(test_episode_range):
            test_observation = test_env.reset()
            test_time_step = 0
            test_score = 0
            while test_time_step < 50:
                test_obs = np.concatenate([test_observation['observation'],test_observation['achieved_goal'],test_observation['desired_goal']],axis=-1,dtype=np.float32) #  ili axis = 0
                #obs_tensor = torch.from_numpy(obs).to(agent.actor.device)
                test_action = agent.choose_action(test_obs)
                test_next_observation, test_reward, test_done = test_env.step(test_action)
                test_observation = test_next_observation
                
                test_score +=  test_reward
                if test_done:
                    break
                test_time_step += 1
                time.sleep(1./30)
            test_scores.append(test_score)
        agent.training_mode()
        test_env.close()
        print(f"Test Episode - score: {test_score}, average score: {np.average(test_scores[-100:])}")




agent = Agent(lr_actor,lr_critic,input_dims,obs_dims,n_actions,max_action,fc1_dim=512,fc2_dim=512,batch_size=2048)

#input dims bi trebao biti 13
#n_actions bi trebao biti 3

#input za actora je 13
#input za critica je 16

#env.make_goal(action)
#env._load_golf_ball()
#time.sleep(.5)
episode_length = 50
num_of_episodes = 100000
scores = []
actor_losses = []
critic_losses = []
temperature_losses = []
loss = []
episodes = []



observations = []
achieved_goals=[]
desired_goals =[]
actions = []
rewards = []
next_observations = []
next_achieved_goals=[]
next_desired_goals =[]



#plt.ioff()

for episode in trange(num_of_episodes):
   
        observation = env.reset()
        time_step = 0
        episodes.append(episode)
        
        while time_step < episode_length:
            obs = np.concatenate([observation['observation'],observation['achieved_goal'],observation['desired_goal']],axis=-1,dtype=np.float32) #  ili axis = 0
            #obs_tensor = torch.from_numpy(obs).to(agent.actor.device)
            action = agent.choose_action(obs)
            next_observation, reward, done = env.step(action)

            agent.memory.real_buffer.append(observation,
                                            action,
                                            reward,
                                            done,
                                            next_observation,
                                            1)
            # agent.memory.episode_buffer.append(observation,
            #                                 action,
            #                                 reward,
            #                                 done,
            #                                 next_observation,
            #                                 1)
            observations.append(observation['observation'])
            achieved_goals.append(observation['achieved_goal'])
            desired_goals.append(observation['desired_goal'])
            actions.append(action)
            rewards.append(reward)
            next_observations.append(next_observation['observation'])
            next_achieved_goals.append(next_observation['achieved_goal'])
            next_desired_goals.append(next_observation['desired_goal'])
            
            
            if episode > 30:
                batch = agent.memory.sample()
                #agent.learn(real_batch)
                #agent.learn(her_batch)
                #combined_batch = {key: np.concatenate([real_batch[key], her_batch[key]], axis=0) for key in real_batch}
                agent.learn(batch)
               
            
            if done:
                break
            time_step += 1
            observation = next_observation

        score = np.sum(rewards)
        scores.append(score)
        print(f"Episode:, {episode}, score: {score}, average score: {np.mean(scores[-100:],)}")
        if (episode + 1 )% 50 == 0:
            test()
        """
        critic_losses = []

        her_observations = agent.memory.episode_buffer.observations
        her_achieved_goals = agent.memory.episode_buffer.achieved_goals
        her_desired_goals = agent.memory.episode_buffer.desired_goals

        her_obs = {
            'observation': her_observations,
            'achieved_goal': her_achieved_goals,
            'desired_goal': her_desired_goals
        }
        her_actions = agent.memory.episode_buffer.actions
        her_next_observations = agent.memory.episode_buffer.next_observations
        her_next_achieved_goals = agent.memory.episode_buffer.next_achieved_goals
        her_next_desired_goals = agent.memory.episode_buffer.next_desired_goals

        her_next_obs = {
            'observation': her_next_observations,
            'achieved_goal': her_next_achieved_goals,
            'desired_goal': her_next_desired_goals
        }



        end_achieved_episode_goal = observation['achieved_goal']
        # das li ce raditi posto je jedan parametar np.array oblika(time_step, goal_dim) a drugi samo np.array(goal_dim)
        her_rewards = env.compute_reward(her_obs['achieved_goal'],end_achieved_episode_goal)
        her_dones = her_rewards + 1 #ish
        agent.memory.her_buffer.append(her_obs,
                                            her_actions,
                                            her_rewards,
                                            her_dones,
                                            her_next_obs,
                                            time_step)
    
        agent.memory.episode_buffer.reset_buffer()
        """
        end_achieved_episode_goal = achieved_goals[-1]
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
            
            
            # das li ce raditi posto je jedan parametar np.array oblika(time_step, goal_dim) a drugi samo np.array(goal_dim)
            her_reward = env.compute_reward(achieved_goals[index],end_achieved_episode_goal)
            her_done = her_reward + 1 #ish
            agent.memory.her_buffer.append(her_obs,
                                                actions[index],
                                                her_reward,
                                                her_done,
                                                her_next_obs,
                                                1)





        #if episode > 1000 or np.mean(scores[-100:]) > -25 : #and np.mean(scores[-100:]) > -30:
        #    env.phase =1
  
        if np.mean(scores[-10:]) > -25 :
            env.difficulty = min(env.difficulty+0.05, 1.1)
        elif np.mean(scores[-10:]) < -48 :
            env.difficulty = max(env.difficulty-0.05, 0.6)


        observations = []
        achieved_goals=[]
        desired_goals =[]
        actions = []
        rewards = []
        next_observations = []
        next_achieved_goals=[]
        next_desired_goals =[]
        
        #print(action)


time.sleep(122)