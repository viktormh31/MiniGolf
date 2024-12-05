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
        """
        observation = env.reset()
        
            while time_step < episode_length:
                obs = np.concatenate([observation['observation'],observation['achieved_goal'],observation['desired_goal']],axis=-1 ili axis = 0)
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
                
                if episode > 10:
                    real_batch, her_batch = agent.memory.sample()
                    agent.learn(real_batch)
                    agent.learn(her_batch)

                if done:
                    break
                time_step += 1
            score = np.sum(agent.memory.episode_buffer.rewards)
            
            her_observations = agent.memory.episode_buffer.observations
            her_action = agent.memory.episode_buffer.actions
            her_next_observations = agent.memory.episode_buffer.next_observations


            end_achieved_episode_goal = observation['achieved_goal']
            her_reward = env.compute_reward(agent.memory.episode_buffer.observation['achieved_goal'],end_achieved_episode_goal)
            her_done = bool(her_reward + 1) ish
            agent.memory.her_buffer.append(her_observation,
                                                her_action,
                                                her_reward,
                                                her_done,
                                                her_next_observation,
                                                time_step)
        
            agent.memory.episode_buffer.reset()
        """

        if done:
            break


        time.sleep(2/120)
        #print(action)
time.sleep(122)