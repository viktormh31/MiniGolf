import os
#import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
from memory import HerBuffer
#import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from ultralytics import YOLO
import cv2 as cv
import time

#https://arxiv.org/pdf/1812.05905 - SAC with automatic entropy adjustment


DEVICE = 'cuda'

class CriticNetwork(nn.Module):
    def __init__(self,input_dims, n_actions, fc1_dims, fc2_dims, lr_critic,name):
        super(CriticNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc2_dims
        self.lr_actor = lr_critic

        self.fc1 = nn.Sequential(
                nn.Linear(self.input_dims + self.n_actions, self.fc1_dims),
                nn.ReLU()
            )

        self.fc2 = nn.Sequential(
                nn.Linear(self.fc1_dims,self.fc2_dims),
                nn.ReLU()
            )
        self.fc3 = nn.Sequential(
                nn.Linear(self.fc2_dims,self.fc3_dims),
                nn.ReLU()
            )
        self.q = nn.Linear(self.fc3_dims,1)

        self.optimizer = optim.AdamW(self.parameters(),lr=lr_critic)
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')

        
        self.checkpoint_file = os.path.join('models/agent/checkpoint',f"{name}.pth")

        self.to(self.device)

    def forward(self,state,action):
        x = torch.concatenate((state,action), dim=1)
        action_value = self.fc1(x)
        action_value = self.fc2(action_value)
        action_value = self.fc3(action_value)
        q = self.q(action_value)

        return q

class ActorNetwork(nn.Module):
    def __init__(self,input_dims, n_actions, max_action, fc1_dims, fc2_dims, lr_actor):
        super(ActorNetwork,self).__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.max_action = max_action
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.fc3_dims = fc2_dims
        self.lr_actor = lr_actor
        self.reparam_noise = float(1e-6)
        

        self.fc1 = nn.Sequential(
                nn.Linear(self.input_dims, self.fc1_dims),
                nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Linear(self.fc1_dims,self.fc2_dims),
            nn.ReLU()
        )
        self.fc3 = nn.Sequential(
            nn.Linear(self.fc2_dims,self.fc3_dims),
            nn.ReLU()
        )
        self.mu = nn.Linear(self.fc3_dims,self.n_actions)
        self.var = nn.Linear(self.fc3_dims,self.n_actions)
            
        self.optimizer = optim.AdamW(self.parameters(),lr=lr_actor)
        self.device = torch.device(DEVICE if torch.cuda.is_available() else 'cpu')
        
        self.checkpoint_file = os.path.join('models/agent/checkpoint', "actor.pth")

        
        self.to(self.device)
        
        
    def forward(self,state):
        prob = self.fc1(state)
        prob = self.fc2(prob)
        prob = self.fc3(prob)
        mu = self.mu(prob)
        log_var = self.var(prob)
        #var = torch.clamp(log_var,min=self.reparam_noise, max=1)
        log_var = torch.clamp(log_var,min=-20, max=2)
        return mu, log_var


    def sample(self,state,reparametrization = True):
        mu, log_var = self.forward(state)
        var = log_var.exp()
        probabilities = torch.distributions.Normal(mu,var)

        if reparametrization:
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()
        
        action = torch.tanh(actions) * torch.tensor(self.max_action).to(self.device)
        log_probs = probabilities.log_prob(actions)
        log_probs -= torch.log(1-action.pow(2)+self.reparam_noise)
        log_probs = log_probs.sum(-1, keepdim=True)
    

        return action, log_probs
    

class Agent(object):
    def __init__(self, lr_actor, lr_critic, input_dims,obs_dims, n_actions
                 ,max_action,tau = 0.005, gamma= 0.99, max_memory_size= 1000000
                 , fc1_dim=256, fc2_dim=256, batch_size=256, batch_ratio = .5, reward_scale=2):
        
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.obs_dims = obs_dims
        self.max_action = max_action
        self.tau = tau
        self.gamma = gamma
        self.max_memory_size = max_memory_size
        self.batch_size = batch_size
        self.batch_ratio = batch_ratio
        self.reward_scale = reward_scale

        self.actor_losses = []
        self.critic_losses = []
        self.temperature_losses = []
        


        self.actor = ActorNetwork(self.input_dims,self.n_actions,self.max_action, fc1_dim,fc2_dim,lr_actor)
        self.critic_1 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic,"critic_1")
        self.target_critic_1 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic,"target_critic_1")
        self.critic_2 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic,"critic_2")
        self.target_critic_2 = CriticNetwork(self.input_dims,self.n_actions,fc1_dim, fc2_dim,lr_critic,"target_critic_2")


        self.target_entropy = -2 #self.n_actions
        self.temperature = 0.3
        self.log_temperature = torch.zeros(1, requires_grad=True, device= DEVICE)
        self.temperature_optimizer = optim.AdamW(params=[self.log_temperature],lr=lr_actor)

        self.initialize_weights(self.actor)
        self.initialize_weights(self.critic_1)
        self.initialize_weights(self.critic_2)
        
        self.model = YOLO('models/golf_recognition_yolov8.pt').to(self.actor.device)
        self.picture_height = 640
        self.picture_width = 640

        self.last_position = None
        self.last_time = None

    
        self.update_network_params(1)
        self.memory = HerBuffer(self.batch_size, self.batch_ratio,  50, self.obs_dims, self.n_actions,
                                3, self.max_memory_size)
        
    def update_network_params(self, tau= None):
        if tau is None:
            tau = self.tau

        for eval_param, target_param in zip(self.critic_1.parameters(), self.target_critic_1.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
        for eval_param, target_param in zip(self.critic_2.parameters(), self.target_critic_2.parameters()):
            target_param.data.copy_(tau*eval_param + (1.0-tau)*target_param.data)
            
    def initialize_weights(self,model):
        for layer in model.modules():
            if isinstance(layer, nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def learn(self,batch):

        observations_tensor = batch['observations']
        achieved_goals_tensor = batch['achieved_goals']
        desired_goals_tensor = batch['desired_goals']
        actions_tensor = batch['actions']
        rewards_tensor = batch['rewards']
        dones_tensor = batch['dones']
        next_observations_tensor = batch['next_observations']
        next_achieved_goals_tensor = batch['next_achieved_goals']
        next_desired_goals_tensor = batch['next_desired_goals']
    
        obs = torch.concat((observations_tensor, achieved_goals_tensor, desired_goals_tensor),dim=1)
        obs_ = torch.concat((next_observations_tensor, next_achieved_goals_tensor, next_desired_goals_tensor),dim=1)

        #-------Critic networks update-------#

        old_critic_values_1 = self.critic_1.forward(obs,actions_tensor).squeeze()
        old_critic_values_2 = self.critic_2.forward(obs,actions_tensor).squeeze()
        with torch.no_grad():
            new_actions, log_probs = self.actor.sample(obs_,reparametrization=False)
            log_probs = log_probs.view(-1)


            target_values_next_states_1 = self.target_critic_1.forward(obs_,new_actions).squeeze()
            target_values_next_states_2 = self.target_critic_2.forward(obs_,new_actions).squeeze() 
            target_values_next_states = torch.min(target_values_next_states_1,target_values_next_states_2)  - self.temperature* log_probs
            
            q_hat = rewards_tensor + self.gamma*(1-dones_tensor)*(target_values_next_states) # target_values_next_states and without temp*log_probs if 2 critics
           
        critic_loss_1 = F.mse_loss(old_critic_values_1,q_hat)
        critic_loss_2 = F.mse_loss(old_critic_values_2,q_hat)
        critic_loss = critic_loss_1 + critic_loss_2

        self.critic_1.optimizer.zero_grad()
        self.critic_2.optimizer.zero_grad()
        critic_loss.backward()
        self.critic_1.optimizer.step()
        self.critic_2.optimizer.step()

        #-------Actor network update-------#
        new_actions, log_probs = self.actor.sample(obs,reparametrization=True)
        log_probs = log_probs.view(-1)
        critic_values_1 = self.critic_1.forward(obs,new_actions)
        critic_values_2 = self.critic_2.forward(obs,new_actions)
        critic_values = torch.min(critic_values_1,critic_values_2).squeeze()

        log_probs_temp = self.temperature * log_probs
    
        actor_loss = log_probs_temp - critic_values
        actor_loss = torch.mean(actor_loss)
        self.actor.optimizer.zero_grad()
        actor_loss.backward()
        self.actor.optimizer.step()

        #-------Temperature network update-------#
      
        log_probs = log_probs.detach()
        temperature_loss =-1*(self.log_temperature *(log_probs + self.target_entropy)).mean()
        self.temperature_optimizer.zero_grad()
        temperature_loss.backward()
        self.temperature_optimizer.step()
        self.temperature = torch.exp(self.log_temperature)

        self.update_network_params()

        return actor_loss,critic_loss,temperature_loss, self.temperature, self.log_temperature

    def evaluate_mode(self):
        self.actor.eval()

    def training_mode(self):
        self.actor.train()

    def choose_action(self,obs,time_step,reparametrization = False):

        processed_obs = self.process_observation(obs,time_step)
        state = torch.squeeze(torch.concat([processed_obs['observation'],processed_obs['achieved_goal'],processed_obs['desired_goal']],dim=-1)).to(self.actor.device) #,dtype=np.float32)

        actions, _ = self.actor.sample(state,reparametrization)

        return actions
   
    def process_observation(self,obs,time_step):

        tcp_pos = obs['robot_obs']['tcp_pos'].to(self.actor.device)
        tcp_rot = obs['robot_obs']['tcp_rot'].to(self.actor.device)
        tcp_vel = obs['robot_obs']['tcp_vel'].to(self.actor.device)
        bgr_image = obs['camera_image']
        golf_ball_pos, golf_hole_pos, golf_ball_speed = self.process_image(bgr_image,time_step)

        observation = torch.concat((
                     tcp_pos,tcp_rot, tcp_vel,golf_ball_speed,
         ),dim= -1).to(device=self.actor.device)
        

        return {
            'observation': observation,
            'achieved_goal': golf_ball_pos,
            'desired_goal': golf_hole_pos
        }                                               
        
    def process_image(self, img,time_step):
        """
        label_number :
            - 0.0  --> golf ball
            - 1.0  --> golf hole

        !!!!! ubaci logiku ako ne pronadju u datom framu loptu ili rupu koje koordinate da im zada! !!!!!
        """
       # bgr_image = np.reshape(img[2], (self.picture_height, self.picture_width, -1))[:, :, :3]  # Extract RGB
        ball_found = False
        hole_found = False

        rgb_image = cv.cvtColor(img,cv.COLOR_BGR2RGB)
        start = time.time()
        results = self.model(rgb_image, verbose = False)
        end = time.time()
        processing_time = start - end
        #print(1000*(end - start))
        #img = results[0].plot()
        #cv.imshow("Live Tracking",rgb_image)
        #cv.waitKey(1)
        for r in results:
            #print(r.boxes)
            for detected_object in r.boxes.data:
                #print(detected_object)
                label_number = detected_object[-1]
                coords = detected_object[:4]
                real_coords = self.get_coords_from_yolo(coords)

                if label_number == 0.0:
                    golf_ball_coords = real_coords
                    ball_found = True
                elif label_number == 1.0:
                    golf_hole_coords = real_coords
                    hole_found = True

        #cv.imshow("Live Tracking", rgb_image)


        if not ball_found:
            golf_ball_coords = torch.squeeze(self.memory.real_buffer.return_achieved_goals(1))
        else:
            golf_ball_coords = torch.cat((golf_ball_coords,torch.tensor([0.02],device=self.actor.device)))

        if not hole_found:
            golf_hole_coords = torch.squeeze(self.memory.real_buffer.return_desired_goals(1))
        else:
            golf_hole_coords = torch.cat((golf_hole_coords,torch.tensor([0.02], device= self.actor.device)))
        

        golf_ball_speed = self.calculate_speed(golf_ball_coords, time_step)


        return golf_ball_coords, golf_hole_coords, golf_ball_speed
    
    def calculate_speed(self,current_pos, current_step):
        if self.last_position is None or self.last_time is None:
            speed = torch.tensor([0,0,0], device= self.actor.device)
        else:
            delta_pos = current_pos- self.last_position
            delta_t = current_step - self.last_time
            speed = (delta_pos / delta_t) if delta_t > 0 else torch.tensor([0,0,0]).to(self.actor.device)

        

        return speed

    def get_coords_from_yolo(self,yolo_coords):
        y1,x1,y2,x2 = yolo_coords
        real_x1 = (x1 + x2)*0.92376/(2*640) + 0.3381
        real_y1 = (y1 + y2)*0.92376/(2*640) -0.46188

        return torch.tensor([real_x1.round(decimals=4),real_y1.round(decimals=4)]).to(self.actor.device)
    
    def real_memory_append(self,obs,action,reward,done,next_obs,time_step,size_of_append):
        
        processed_obs = self.process_observation(obs,time_step)
        self.last_position = processed_obs['achieved_goal']
        self.last_time = time_step
        processed_next_obs = self.process_observation(next_obs,time_step + 1)
        self.memory.real_buffer.append(processed_obs,action,reward,done,processed_next_obs,size_of_append)

        return 0

    def her_memory_append(self,episode_length,her_rewards,her_dones):
        """
        Prosledjujemo duzinu epizode: episode_length, rewards i dones dobijene od env-a.
        Uzimamo poslednjih (episode_length) time stepova iz  real_buffera, pravimo her_desired_goals
        tensor odgovarajuceg oblika, menjamo desired_goals iz realbuffera i to appendujemo na her_buffer
        
        
        """
        episode_params = self.memory.real_buffer.return_episode(episode_length)

        her_desired_goal = self.memory.real_buffer.return_achieved_goals(1)
        her_desired_goals = her_desired_goal.repeat(episode_length,1)

        her_obs = {
                    'observation': episode_params['obs']['observation'],
                    'achieved_goal': episode_params['obs']['achieved_goal'],
                    'desired_goal': her_desired_goals,
        }

        her_next_obs = {
                    'observation': episode_params['next_obs']['next_observation'],
                    'achieved_goal': episode_params['next_obs']['next_achieved_goal'],
                    'desired_goal': her_desired_goals,
        }

        self.memory.her_buffer.append(her_obs,
                                      episode_params['actions'],
                                      her_rewards,
                                      her_dones,
                                      her_next_obs,
                                      episode_length)

    def save_models(self, suffix=""):
        print('... saving models ...')

        torch.save(self.actor.state_dict(), self._with_suffix(self.actor.checkpoint_file, suffix))
        torch.save(self.critic_1.state_dict(), self._with_suffix(self.critic_1.checkpoint_file, suffix))
        torch.save(self.critic_2.state_dict(), self._with_suffix(self.critic_2.checkpoint_file, suffix))
        torch.save(self.target_critic_1.state_dict(), self._with_suffix(self.target_critic_1.checkpoint_file, suffix))
        torch.save(self.target_critic_2.state_dict(), self._with_suffix(self.target_critic_2.checkpoint_file, suffix))

    def load_models(self, suffix=""):
        print('... loading models ...')

        self.actor.load_state_dict(torch.load(self._with_suffix(self.actor.checkpoint_file, suffix)))
        self.critic_1.load_state_dict(torch.load(self._with_suffix(self.critic_1.checkpoint_file, suffix)))
        self.critic_2.load_state_dict(torch.load(self._with_suffix(self.critic_2.checkpoint_file, suffix)))
        self.target_critic_1.load_state_dict(torch.load(self._with_suffix(self.target_critic_1.checkpoint_file, suffix)))
        self.target_critic_2.load_state_dict(torch.load(self._with_suffix(self.target_critic_2.checkpoint_file, suffix)))

    def _with_suffix(self, path, suffix):
        """Append suffix to checkpoint filename before the .pth extension."""
        if suffix:
            return path.replace(".pth", f"_episode_{suffix}.pth")
        return path