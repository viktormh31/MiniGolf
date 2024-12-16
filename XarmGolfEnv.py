import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import gymnasium as gym
from gymnasium import spaces

class XarmRobotGolf():

    def __init__(self, config):
        self.time_step = 1./240
        

      
        
        
       
        # env params
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.pos_space = spaces.Box(low=np.array([0.4,-0.4,0.02]), high=np.array([0.8,0.4,0.02]))
        self.dt = self.time_step * 40
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        self.phase = 0

        # bullet setup
        if config['GUI']:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT) 
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0,0.0,-9.81)
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(cameraDistance=1, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])


         # load plane
        self._load_plane()


        # robot parameters
        self.num_joints = 17
        self.gripper_driver_index = 10
        self.gripper_base_index = 9
        self.arm_eef_index = 8
        self.tcp_index = 16
        self.reward_type = config['reward_type']
        self.start_pos = [0,0,0]
        self.start_orientation = p.getQuaternionFromEuler([0,0,0])
        self.gripper_base_default_pos = [0.40, 0., 0.02]
        self.max_vel = 1

        # load robot
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7.urdf')
        self.xarm = p.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)


        # hole params
        self.hole_space = spaces.Box(low=np.array([0.8, -0.25, 0.0]),high=np.array([0.8, 0.25, 0.0]))
        self.hole_default_pos = np.array([0.8,0.0,0.0])
        self.distance_threshold = 0.05

        # load goal
        self._load_golf_hole()
        

        # ball params
        self.golf_ball_default_pos = [0.45,0,0.02]
        self.golf_ball_pos = self.golf_ball_default_pos
        
        # load ball
        self._load_golf_ball()


      
     
    #  Basic methods
    # ---------------

    def step(self,action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        p.stepSimulation()
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'],obs['desired_goal'])
        done = reward + 1
        time.sleep(self.time_step)
        return obs, reward, done, #info
        
    def reset(self):
        #return obs
        self._reset_sim()
        #self.goal = self._sample_goal()
        if self.phase == 0:
            self._sample_goal(False)
        elif self.phase == 1:
            self._sample_goal(True)

        return self._get_obs()

    def compute_reward(self, achieved_goal, goal):
        distance = np.linalg.norm(achieved_goal - goal, axis=-1)

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

        tcp_state = p.getLinkState(self.xarm, self.tcp_index, computeLinkVelocity=1)
        tcp_pos = np.array(tcp_state[0])
        tcp_vel = np.array(tcp_state[6])

        #golf ball state
        self.golf_ball_pos, _ = p.getBasePositionAndOrientation(self.golf_ball)
        ball_vel = p.getBaseVelocity(self.golf_ball)[0]
        # distance
        distance = np.array([np.linalg.norm(self.hole_pos-self.golf_ball_pos,axis=-1)])

        # observation
        obs = np.concatenate((
                    tcp_pos, tcp_vel, self.golf_ball_pos, ball_vel, distance
        ),axis= -1)
        #obs = np.concatenate((obs,distance),axis =-1)

        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(self.golf_ball_pos),
            'desired_goal': self.hole_pos.copy()
        }

       
    def _reset_sim(self):

        self._reset_robot_arm()
        self._reset_golf_ball()
        
    
    def _load_golf_ball(self):
            # implement bool for picking default position or random position,
            # add in __init__() self.golf_ball_space = [x,y,z] do [o,p,q]
        ball_shape = p.createVisualShape(shapeType= p.GEOM_SPHERE, radius= 0.02, rgbaColor = [0, 0 ,0.8, 1])
        ball_colision = p.createCollisionShape(shapeType = p.GEOM_SPHERE, radius = 0.02)

        self.golf_ball = p.createMultiBody(baseMass = 0.1,
                                    baseInertialFramePosition = [0,0,0],
                                    baseCollisionShapeIndex = ball_colision,
                                    baseVisualShapeIndex = ball_shape,
                                    basePosition = self.golf_ball_default_pos,
                                    useMaximalCoordinates = True)

    def _load_plane(self):
        plane = p.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])
        p.changeDynamics(plane,-1, 
                 lateralFriction = .05,
                 rollingFriction = .1,
                 restitution = .7)
        
    def _load_golf_hole(self):
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_golf_hole.urdf')
        self.hole = p.loadURDF(fullpath,self.hole_default_pos,self.start_orientation, useFixedBase=True)
        self.hole_pos = self.hole_default_pos

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
    
    def _reset_golf_ball(self):
        p.resetBasePositionAndOrientation(self.golf_ball, self.golf_ball_default_pos, self.start_orientation)
    
    def _reset_golf_hole(self):
        p.resetBasePositionAndOrientation(self.hole,self.hole_default_pos,self.start_orientation)
        
    def setup_golf_course(self):
        
        raise NotImplementedError
    

    #  Goal methods
    # --------------
    def _sample_goal(self,rand_pos):
        if rand_pos == False:
            self.hole_pos = self.hole_default_pos
        else:
            self.hole_pos = np.array(self.hole_space.sample())
        p.resetBasePositionAndOrientation(self.hole, self.hole_pos, self.startOrientation)


    def make_goal(self,point_position):

    
        p.addUserDebugLine(point_position, [point_position[0], point_position[1], point_position[2] + 0.01], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0], point_position[1], point_position[2] - 0.01], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] - 0.01, point_position[1], point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] + 0.01, point_position[1], point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0], point_position[1] + 0.01, point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] , point_position[1]- 0.01, point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)





