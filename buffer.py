import numpy as np
import random
from collections import deque

class HERBuffer():

    def __init__(self, odnos = 0.5, batch_size = 256, buffer_len = int(1e6)):

        self.her_batch = int(odnos * batch_size)
        self.batch = int((1-odnos) * batch_size)
        self.buffer_len = buffer_len
        
        # instanciras redovan buffer
        # instanciras HER buffer
        self.her_buffer = ReplayBuffer(buffer_len, self.her_batch)
        self.buffer = ReplayBuffer(buffer_len, self.batch)
        # cuvanje cele epizode
        #self.states = []
        self.observations = {
            'observations':[],
            'achieved_goals':[],
            'desired_goals':[],
        }
        self.actions  = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        
        #self.goals = []


    def push(self, *transition):       # ----------------------------zbog čega "*" ???
        #state,action, reward,next_state, done, goal = transition
        observation,action, reward,next_state, done = transition

        for key in observation:
            self.observations[key].append(observation[key])
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.next_states.append(next_state)
        
        #self.goals.append(goal)
        
        self.buffer.append(observation, action, reward, next_state)#, done, goal
        pass

    def reset_episode(self):
        
        #self.states = []
        self.observations = []
        self.actions  = []
        self.rewards = []
        self.next_states = []
        self.dones = []
       
        #self.goals = []
        
        #return new_rewards
        pass

    def sample(self):
        buffer_sample = self.buffer.sample()
        her_buffer_sample = self.her_buffer.sample()
        
        
      
        return buffer_sample, her_buffer_sample
        # sample redovan buffer
        # sample HER buffer
        #torch.cat() axis = 0
        #np.concatenate

    def return_episode(self):

        #states = np.array()
    
        actions = np.array(self.actions)
        next_states = np.array(self.next_states)
        
        #her_rewards = rewww(goal_from_states,her_goal)
        #her_rewards = np.reshape(her_rewards,(time_step,1))
        #dones = her_rewards.copy()
        #dones = np.add(dones,1)
        return self.observations, actions, next_states
    
    

class ReplayBuffer(object):
    def __init__(self, max_size,batch_size):
        self.mem_size = max_size
        self.batch_size = batch_size
        self.counter = 0
        #self.state_memory = deque([],maxlen=self.mem_size)
        self.observation_memory = deque([],self.mem_size)
        self.next_state_memory = deque([],self.mem_size)
        self.reward_memory = deque([],self.mem_size)
        self.done_memory = deque([],self.mem_size)
        self.action_memory = deque([],self.mem_size)
        
        #self.goal_memory = deque([],self.mem_size)
        # self.state_memory = []
        # self.next_state_memory = []
        # self.reward_memory = []
        # self.done_memory = []
        # self.action_memory = []
        # self.goal_memory = []
        
        
        
    #def append(self,state,action,reward,next_state, done,goal):
    def append(self,observation,action,reward,next_state, done):#,goal
        if len(action.shape) == 1:
            #self.state_memory.append(state)
            for key in observation:
                self.observation_memory[key].append(observation[key])
            self.next_state_memory.append(next_state)
            self.action_memory.append(action)
            self.reward_memory.append(reward)
            self.done_memory.append(done)
            #self.goal_memory.append(goal)
            
           
            
        else:
            #self.state_memory.extend(state)
            for key in observation:
                self.observation_memory[key].extend(observation[key])
            self.next_state_memory.extend(next_state)
            self.action_memory.extend(action)
            self.reward_memory.extend(reward)
            self.done_memory.extend(done)
            #self.goal_memory.extend(goal*len(state)) ## zbog istog goal u funkciji append 
                                                     #za her buffer mi ubaci samo jednom goal, 
                                                     #pa ne bude odgovarajuće  dimenzije
           
        
    def sample(self):
        max_memory = len(self.action_memory)
        
        batch = np.array(random.sample(range(max_memory), self.batch_size), dtype=np.int32)
      
       
        #states =  [self.state_memory[i] for i in batch]
        observations = []
        next_states = [self.next_state_memory[i] for i in batch]
        actions = [self.action_memory[i] for i in batch]
        rewards = [self.reward_memory[i] for i in batch]
        dones = [self.done_memory[i] for i in batch]
        goals = [self.goal_memory[i] for i in batch]
        
        #primeni buffer na liste umesto deque
        #vector gymnasium
        #tqdm
        
        
        return observations, actions, rewards, next_states, dones, goals
        #return states, actions, rewards, next_states, dones, goals
