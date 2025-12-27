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
    id= 'test42 - camera test 7 - pos + vel, changed camera position - Final test long'
)  
env =XarmRobotGolf(config)
#test_env = XarmRobotGolf(test_config)
lr_actor = 0.0001
lr_critic = 0.003
input_dims = 19
obs_dims = 13
n_actions = 4
max_action = 1

agent = Agent(lr_actor,lr_critic,input_dims,obs_dims,n_actions,max_action,fc1_dim=512,fc2_dim=512,batch_size=2048,gamma=0.95)


test_scores = []
phase_average = []
test_episode_count = 0

#@torch.no_grad
def test(episode_reached):
    env.close()
    with torch.no_grad():
        test_env = XarmRobotGolf(test_config)
        test_env.phase = env.phase
        test_env.reset() #unnecessary probablly
        agent.evaluate_mode()
        test_episode_range = 5
        made = 0.
        for test_episode in trange(test_episode_range):
            test_observation = test_env.reset()
            test_time_step = 0
            test_score = 0
            agent.last_position = None
            agent.last_time = None

            while test_time_step < episode_length:
                action = agent.choose_action(test_observation, test_time_step,False)
                test_next_observation, test_reward, test_done, _ = test_env.step(action)
            
                test_score +=  test_reward
                test_time_step += 1
                if test_done:
                    break
                
                test_observation = test_next_observation
               
                
            test_scores.append(test_score)

            if test_score> -50:
                made+=1.

            print(f"Test Episode {len(test_scores)}, score - {test_score}")

        agent.training_mode()
        test_env.close()
        print(f"Test 200 average score: {np.average(test_scores[-200:])}")
        print(f"Success rate - {made/2.}%")
        phase_average.append(np.average(test_scores[-200:]))
        env.connect()
        agent.save_models(episode_reached)


episode_length = 50
num_of_episodes = 30000  
scores = []

als = []
cls = []
tls = []
success= []

def main():
    test(10000)
    for episode in trange(num_of_episodes):
            observation = env.reset()
            time_step = 0
            episode_score = 0
            agent.last_position = None
            agent.last_time = None

            
            while time_step < episode_length:
                
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
                    batch = agent.memory.sample()
                    al, cl, tl ,alpha,log_alpha= agent.learn(batch)
                
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
            
            her_achieved_goals= agent.memory.real_buffer.return_achieved_goals(time_step)   # napravi funkciju za ovo
            her_rewards, her_dones = env.compute_her_reward(her_achieved_goals[-1], her_achieved_goals)
            agent.her_memory_append(time_step,her_rewards,her_dones)

            if episode == 3000:
                success_rate = sum(1 for x in scores[-200:] if x > -50)/2.
                success.append(success_rate)
                agent.save_models(episode)
                env.phase = 2  
            elif episode == 10000:
                success_rate = sum(1 for x in scores[-200:] if x > -50)/2.
                success.append(success_rate)
                agent.save_models(episode)
                env.phase = 3
            elif episode % 10000 == 0 and episode > 10000:
                success_rate = sum(1 for x in scores[-200:] if x > -50) / 2. 
                success.append(success_rate)
                agent.save_models(episode)
                #env.phase = 3
        
    wandb.finish()

#agent.load_models(3000)
#test(3000)
main()
