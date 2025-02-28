import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import gymnasium as gym
from gymnasium import spaces

class XarmRobotReach():

    def __init__(self, config):
        self.time_step = 1./240
        

        # robot parameters
        self.distance_threshold = 0.05
        self.num_joints = 17
        self.gripper_driver_index = 10
        self.gripper_base_index = 9
        self.arm_eef_index = 8
        self.tcp_index = 16
        self.reward_type = config['reward_type']
        self.start_pos = [0,0,0]
        self.start_orientation = p.getQuaternionFromEuler([0,0,0])
        # self.joint_init_pos = [0, -0.1, -0.08153217279952825, 
        #                         0.09299669711139864, 1.067692645248743, 0.0004018824370178429, 
        #                         1.1524205092196147, -0.0004991403332530034,0,0,0.9,0.9,0,0.9,0.9] + [0]*2
        self.joint_init_pos = [0, 0.27050686544800806, -0.005340887396375177, -0.2711492861468919
                               , 0.4600086544592818, -0.003261038126179985, 0.46525864138699663
                               , 0.0022592575465344170,0,0.9,0.9,0,0.9,0.9] + [0]*2
        
        
        self.gripper_base_default_pos = [0.40, 0., 0.4]
        self.max_vel = 1
        self.dt = self.time_step * 20
        self.pos_space = spaces.Box(low=np.array([0.2,-0.4,0.0]), high=np.array([0.7,0.4,0.5]))
        self.goal_space = spaces.Box(low=np.array([0.4, -0.15, 0.1]),high=np.array([0.6, 0.15, 0.3]))
        self.goal_default_pos = np.array([0.65,0.0,0.3])
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.phase = 1
        # connect bullet
     
        # p.connect(p.GUI)
        # p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])


        # # bullet setup
        # ###self.seed()
        # p.resetSimulation()
        # p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # p.setGravity(0.0,0.0,-9.81)
        # p.setRealTimeSimulation(0)
        # p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])

        # fullpath = "/home/viktor/miniconda3/envs/minigolf/lib/python3.11/site-packages/pybullet_data/xarm/xarm6_with_gripper.urdf"
        # self.xarm = p.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)
        if config['GUI']:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT) 
           
            
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])

        # bullet setup
        ###self.seed()
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0,0.0,-9.81)
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])
        # load robot
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7.urdf')
        self.xarm = p.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)

        # load plane
        self._load_plane()

        
        # load goal
        self._load_goal()

        # env setup
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        #self.goal = self._sample_goal(False)



        # self.observation_space = spaces.Dict(dict(
        #     observation=spaces.Box(-np.inf, np.inf, shape=obs['observation'].shape, dtype='float32'),
        #     achieved_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        #     desired_goal=spaces.Box(-np.inf, np.inf, shape=obs['achieved_goal'].shape, dtype='float32'),
        # ))

    #  Basic methods
    # ---------------

    def step(self,action):
        
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        p.stepSimulation()
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'],self.goal_pos)
        done = reward + 1
        time.sleep(self.time_step)
        return obs, reward, done, #info
        
    def reset(self):
        #return obs
        self._reset_sim()
        #self.goal = self._sample_goal()
        if self.phase == 0:
            self._reset_goal()
        if self.phase == 1:
            self._sample_goal(True)
        return self._get_obs()

    def compute_reward(self, achieved_goal, goal):
        distance = np.linalg.norm(achieved_goal- goal, axis=-1)

        return (distance < self.distance_threshold).astype(np.float32) - 1.

    #  RobotEnv methods
    # ------------------

    def _set_action(self,action):

        assert action.shape == (3,), 'action shape error'
        current_position = np.array(p.getLinkState(self.xarm,self.tcp_index)[0])
        new_position = current_position + action * self.max_vel * self.dt
        new_position = np.clip(new_position, self.pos_space.low,self.pos_space.high)
        joint_poses = p.calculateInverseKinematics(self.xarm,self.tcp_index,new_position,[1,0,0,0])
        for i in range(1,self.arm_eef_index):
            p.setJointMotorControl2(self.xarm,i,p.POSITION_CONTROL, joint_poses[i-1])
        
    def _get_obs(self):
            #find a way to introduce orientation of gripper as obs

        # robot state
        robot_state = p.getJointStates(self.xarm, np.arange(0,self.num_joints))

        # gripper state
        #gripper_pos = np.array([robot_state[self.gripper_driver_index][0]])
        #gripper_vel = np.array([robot_state[self.gripper_driver_index][1]])
        tcp_state = p.getLinkState(self.xarm, self.tcp_index, computeLinkVelocity=1)
        tcp_pos = np.array(tcp_state[0])
        
        #tcp_vel = np.array(tcp_state[6])

        # goal position
        self.goal_pos, _ = p.getBasePositionAndOrientation(self.goal)

        # distance
        #current_position = np.array(p.getLinkState(self.xarm,self.tcp_index)[0])

        distance = np.array([np.linalg.norm(self.goal_pos-tcp_pos,axis=-1)])

        # observation
        obs = np.concatenate((
                    tcp_pos, distance
        ),axis= -1)
        #obs = np.concatenate((obs,distance),axis =-1)

        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(tcp_pos),
            'desired_goal': self.goal_pos
        }

       
    def _reset_sim(self):

        self._reset_robot()

    def _reset_robot(self):
        tcp_state = p.getLinkState(self.xarm, self.tcp_index, computeLinkVelocity=1)
        tcp_pos = np.array(tcp_state[0])

        for i in range(17):
            p.resetJointState(self.xarm,i,targetValue = self.joint_init_pos[i], targetVelocity = 0)

      
        
    def _load_plane(self):
        plane = p.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])
        p.changeDynamics(plane,-1, 
                 lateralFriction = .05,
                 rollingFriction = .1,
                 restitution = .7)
  

    def _load_goal(self):
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_sphere.urdf')

        self.goal = p.loadURDF(fullpath, self.goal_default_pos, [0,0,0,1],useFixedBase=True)
    # def _reset_robot(self):
    #     for i in range(17):
    #         p.resetJointState(self.xarm,i,targetValue = self.joint_init_pos[i], targetVelocity = 0)

    def _reset_robot_arm(self):
        for _ in range(60): 
            jointPoses = p.calculateInverseKinematics(self.xarm, self.tcp_index, self.gripper_base_default_pos, [1,0,0,0], maxNumIterations = 20)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
            p.setJointMotorControl2(self.xarm, 10, p.POSITION_CONTROL, 1, force=1000)
            p.setJointMotorControl2(self.xarm, 11, p.POSITION_CONTROL, 1, force=1000)
            p.setJointMotorControl2(self.xarm, 13, p.POSITION_CONTROL, 1, force=1000)
            p.setJointMotorControl2(self.xarm, 14, p.POSITION_CONTROL, 1, force=1000)
            p.stepSimulation()
    
    def _reset_goal(self):
        p.resetBasePositionAndOrientation(self.goal, self.goal_default_pos, self.startOrientation)


    def setup_golf_course(self):
        
        raise NotImplementedError
    

    #  Goal methods
    # --------------
    def _sample_goal(self,rand_pos):
        if rand_pos == False:
            self.goal_pos = self.goal_default_pos
        else:
            self.goal_pos = np.array(self.goal_space.sample())
        p.resetBasePositionAndOrientation(self.goal, self.goal_pos, self.startOrientation)
        return self.goal_pos




