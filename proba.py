import time
import numpy as np

from SAC_with_temperature_v2 import Agent
import torch
from XarmGolfEnv import XarmRobotGolf
from XarmReach import XarmRobotReach
from tqdm import trange

#import matplotlib.pyplot as plt
config = {

    'GUI' : True,
    'reward_type' : "sparse",
}
test_config = {
     'GUI' : True,
     'reward_type' : "sparse", 
}


env =XarmRobotGolf(config)

lr_actor = 0.001
lr_critic = 0.001
input_dims = 19
obs_dims = 13
n_actions = 3
max_action = 1

test_scores = []
test_episode_count = 0

def test():
    test_env = XarmRobotEnv(test_config)

    test_episode_range = 1
    for test_episode in range(test_episode_range):
        test_observation = test_env.reset()
        test_time_step = 0
        test_score = 0
        while test_time_step < 50:
            test_obs = np.concatenate([test_observation['observation'],test_observation['achieved_goal'],test_observation['desired_goal']],axis=-1,dtype=np.float32) #  ili axis = 0
            #obs_tensor = torch.from_numpy(obs).to(agent.actor.device)
            test_action = agent.choose_action(test_obs)
            test_next_observation, test_reward, test_done = env.step(test_action)
            test_observation = test_next_observation
            
            test_score +=  test_reward
            if test_done:
                break
            test_time_step += 1
        print(f"Test Episode - score: {test_score}, average score: {np.average(test_scores[:100],)}")
        
    pass
#plt.plot([0, 1, 2], [0, 1, 4])
#plt.show()



agent = Agent(lr_actor,lr_critic,input_dims,obs_dims,n_actions,max_action)

#input dims bi trebao biti 13
#n_actions bi trebao biti 3

#input za actora je 13
#input za critica je 16

#env.make_goal(action)
#env._load_golf_ball()
#time.sleep(.5)
episode_length = 40
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
            
            
            if episode > 100:
                real_batch, her_batch = agent.memory.sample()
                l = agent.learn(real_batch)
                #actor_losses.append(l['actor_loss'].detach().numpy())
                #critic_losses.append(l['critic_loss'].detach().numpy())
                temperature_losses.append(l['temp_loss'].detach().numpy())
                l = agent.learn(her_batch)
                #actor_losses.append(l['actor_loss'].detach().numpy())
                #critic_losses.append(l['critic_loss'].detach().numpy())
                temperature_losses.append(l['temp_loss'].detach().numpy())
            time_step += 1
            if done:
                break
            
            observation = next_observation

        score = np.sum(rewards)
        scores.append(score)
        print(f"Episode:, {episode}, score: {score}, average score: {np.mean(scores[-100:],)}")
        
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





        if episode > 1000 or np.mean(scores[-100:]) > -25 : #and np.mean(scores[-100:]) > -30:
            env.phase =1
  
        

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