import numpy as np
import gymnasium as gym
import gymnasium_robotics
from tqdm import trange
from SAC_with_temperature_v2 import Agent
#gym.register_envs(gymnasium_robotics)
import panda_gym


max_episodes = 10000

lr_actor = 0.001
lr_critic = 0.001
env = gym.make("PandaSlide-v3", render_mode ="human")
observation, _ = env.reset()
obs_dims = 18
input_dims = observation['observation'].shape[0]+observation['achieved_goal'].shape[0] + observation["desired_goal"].shape[0]
n_actions = env.action_space.shape[0]
max_action = 1

test_scores = []
test_episode_count = 0
episode_length = 50
agent = Agent(lr_actor,lr_critic,input_dims,obs_dims,n_actions,max_action)

observations = []
achieved_goals=[]
desired_goals =[]
actions = []
rewards = []
next_observations = []
next_achieved_goals=[]
next_desired_goals =[]




scores = []
def compute_reward(achieved_goal, goal):
    distance = np.linalg.norm(achieved_goal- goal, axis=-1)
    reward = (distance < 0.05).astype(np.float32) - 1.
    return reward



for episode in trange(max_episodes):

    observation, _ = env.reset()

    time_step = -1
    score = 0
    done = False
    
    while time_step < episode_length and not done:
        time_step += 1
        obs = np.concatenate([observation['observation'],observation['achieved_goal'],observation['desired_goal']],axis=-1,dtype=np.float32) #  ili axis = 0

        action = agent.choose_action(obs)
        next_observation, reward,done, trunc, _ = env.step(action)
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
        # observations.append(observation['observation'])
        # achieved_goals.append(observation['achieved_goal'])
        # desired_goals.append(observation['desired_goal'])
        # actions.append(action)
        # rewards.append(reward)
        # next_observations.append(next_observation['observation'])
        # next_achieved_goals.append(next_observation['achieved_goal'])
        # next_desired_goals.append(next_observation['desired_goal'])



        if episode > 10:
            real_batch, her_batch = agent.memory.sample()
            l = agent.learn(real_batch)
            l = agent.learn(her_batch)

        
        
        #if done or trunc:
    
           # break
        
        observation = next_observation

    score = np.sum(agent.memory.episode_buffer.rewards)
    scores.append(score)
    print(f"Episode:, {episode}, score: {score}, average score: {np.mean(scores[-100:])}, time step : {time_step}")
        
    """
    for index in range(time_step):
        her_obs = {
            'observation': observations[index],
            'achieved_goal': achieved_goals[index],
            'desired_goal': desired_goals[index]
        }
        her_next_obs = {
            'observation': next_observations[index],
            'achieved_goal': next_achieved_goals[index],
            'desired_goal': next_desired_goals[index]
        }
        
        end_achieved_episode_goal = achieved_goals[-1]
        # das li ce raditi posto je jedan parametar np.array oblika(time_step, goal_dim) a drugi samo np.array(goal_dim)
        her_reward = compute_reward(achieved_goals[index],end_achieved_episode_goal)
        her_done = her_reward + 1 #ish
        agent.memory.her_buffer.append(her_obs,
                                            actions[index],
                                            her_reward,
                                            her_done,
                                            her_next_obs,
                                            1)



        
    """ 
    if time_step !=1 :
        #score = np.sum(agent.memory.episode_buffer.rewards)
        end_achieved_episode_goal = agent.memory.episode_buffer.achieved_goals[-1]
        
        critic_losses = []
        her_observations = agent.memory.episode_buffer.observations
        her_achieved_goals = agent.memory.episode_buffer.achieved_goals
        her_desired_goals = agent.memory.episode_buffer.desired_goals
        her_obs = {
            'observation': her_observations,
            'achieved_goal': her_achieved_goals,
            'desired_goal': end_achieved_episode_goal
        }

        her_actions = agent.memory.episode_buffer.actions

        her_next_observations = agent.memory.episode_buffer.next_observations
        her_next_achieved_goals = agent.memory.episode_buffer.next_achieved_goals
        her_next_desired_goals = agent.memory.episode_buffer.next_desired_goals
        her_next_obs = {
            'observation': her_next_observations,
            'achieved_goal': her_next_achieved_goals,
            'desired_goal': end_achieved_episode_goal
        }
        
        
        # das li ce raditi posto je jedan parametar np.array oblika(time_step, goal_dim) a drugi samo np.array(goal_dim)
        her_rewards = compute_reward(her_achieved_goals,end_achieved_episode_goal)
        her_dones = her_rewards + 1 #ish
        agent.memory.her_buffer.append(her_obs,
                                            her_actions,
                                            her_rewards,
                                            her_dones,
                                            her_next_obs,
                                            time_step)

        agent.memory.episode_buffer.reset_buffer()
    """ 
    #agent.memory.episode_buffer.reset_buffer()
    observations = []
    achieved_goals=[]
    desired_goals =[]
    actions = []
    rewards = []
    next_observations = []
    next_achieved_goals=[]
    next_desired_goals =[]

"""
