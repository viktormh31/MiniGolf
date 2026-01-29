import numpy as np
import torch
<<<<<<< HEAD
from XarmGolfEnv import XarmRobotGolf

from tqdm import trange
import time
=======
>>>>>>> 68f0900685d43ba07350dabc324f993b85b6c007
import wandb
from tqdm import trange

from SAC_with_temperature_v2 import Agent
from XarmGolfEnv import XarmRobotGolf
from config import (
    TRAIN_ENV_CONFIG,
    TEST_ENV_CONFIG,
    WANDB_CONFIG,
    AGENT_CONFIG,
    TRAINING_CONFIG,
    PHASE_SWITCH
)


wandb.init(
    project=WANDB_CONFIG["project"],
    config=WANDB_CONFIG["hyperparameters"],
    id=WANDB_CONFIG["run_id"]
)

env =XarmRobotGolf(TRAIN_ENV_CONFIG)
agent = Agent(**AGENT_CONFIG)
episode_length = TRAINING_CONFIG["episode_length"]
warm_up_phase = TRAINING_CONFIG["warm_up_phase"]
num_of_episodes = TRAINING_CONFIG["num_episodes"]

test_scores = []
phase_average = []
test_episode_count = 0

episode_scores = []
success = []

def main():
    for episode in trange(num_of_episodes):
            observation = env.reset()
            agent.reset_positions()

            time_step = 0
            episode_score = 0

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
                
                if episode > warm_up_phase:
                    batch = agent.memory.sample()
                    agent.learn(batch)
                
                episode_score += reward
                time_step += 1
                observation = next_observation

                if done:
                    break
                
                
            episode_scores.append(episode_score)
            avg_score = np.mean(episode_scores[-100:])
            print(f"Episode:, {episode}, score: {episode_score}, average score: {avg_score}")
            
            if episode > warm_up_phase:
                wandb.log({"score": avg_score})
            
            her_achieved_goals= agent.memory.real_buffer.return_achieved_goals(time_step)   # napravi funkciju za ovo
            her_rewards, her_dones = env.compute_her_reward(her_achieved_goals[-1], her_achieved_goals)
            agent.her_memory_append(time_step,her_rewards,her_dones)

            if episode in PHASE_SWITCH:
                log_and_save(episode, episode_scores, success)
                env.phase = PHASE_SWITCH[episode]
            elif episode > 10000 and episode % 10000 == 0:
                log_and_save(episode, episode_scores, success, agent)


    wandb.finish()

def log_and_save(episode, scores, success, agent,
                 window = 200, threshold = -50):
    success_rate = sum(x > threshold for x in scores[-window:])/ window * 100
    success.append(success_rate)
    agent.save_models(episode)

#@torch.no_grad
def test(episode_reached):
    env.close()
    with torch.no_grad():
        test_env = XarmRobotGolf(TEST_ENV_CONFIG)
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

#agent.load_models(3000)
#test(3000)
main()
