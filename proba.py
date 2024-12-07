import time
import numpy as np

from SAC_with_temperature_v2 import Agent
import torch
from XarmEnv import XarmRobotEnv
from tqdm import trange
config = {

    'GUI' : True,
    'reward_type' : "sparse",
}
test_config = {
     'GUI' : True,
     'reward_type' : "sparse", 
}


env =XarmRobotEnv(config)

lr_actor = 0.001
lr_critic = 0.001
input_dims = 13
obs_dims = 7
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



agent = Agent(lr_actor,lr_critic,input_dims,obs_dims,n_actions,max_action)

#input dims bi trebao biti 13
#n_actions bi trebao biti 3

#input za actora je 13
#input za critica je 16

#env.make_goal(action)
#env._load_golf_ball()
#time.sleep(.5)
episode_length = 50
num_of_episodes = 5000
scores = []
for episode in trange(num_of_episodes):
   
        observation = env.reset()
        time_step = 0
        
        
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
            agent.memory.episode_buffer.append(observation,
                                            action,
                                            reward,
                                            done,
                                            next_observation,
                                            1)

            observation = next_observation
            
            if episode > 20:
                real_batch, her_batch = agent.memory.sample()
                agent.learn(real_batch)
                agent.learn(her_batch)

            if done:
                break
            time_step += 1
        score = np.sum(agent.memory.episode_buffer.rewards)
        scores.append(score)
        print(f"Episode:, {episode}, score: {score}, average score: {np.average(scores[:100],)}")


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
    

        


        
        #print(action)


time.sleep(122)