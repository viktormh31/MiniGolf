import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import gymnasium as gym
from gymnasium import spaces

class XarmRobotEnv():

    def __init__(self, config):
        self.time_step = 1./60


        # robot parameters
        self.distance_threshold = 0.05
        self.num_joints = 17
        self.gripper_driver_index = 10
        self.gripper_base_index = 9
        self.arm_eef_index = 8
        self.reward_type = config['reward_type']
        self.start_pos = [0,0,0]
        self.start_orientation = p.getQuaternionFromEuler([0,0,0])
        self.joint_init_pos = [0, -0.009068751632859924, -0.08153217279952825, 
                                0.09299669711139864, 1.067692645248743, 0.0004018824370178429, 
                                1.1524205092196147, -0.0004991403332530034,0,0,0.9,0.9,0,0.9,0.9] + [0]*2
        self.gripper_base_default_pos = [0.40, 0., 0.2]
        self.max_vel = 1
        self.dt = self.time_step * 20
        self.pos_space = spaces.Box(low=np.array([0.2,-0.4,0.2]), high=np.array([0.8,0.4,0.6]))
        self.goal_space = spaces.Box(low=np.array([0.6, -0.25, 0.0]),high=np.array([0.9, 0.25, 0.0]))
        self.goal_default_pos = np.array([0.7,0.0,0.0])
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.golf_ball_default_pos = [0.45,0,0.02]
        # connect bullet
     
        p.connect(p.GUI)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])


        # bullet setup
        ###self.seed()
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0.0,0.0,-9.81)
        p.setRealTimeSimulation(0)
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=45, cameraPitch=-10, cameraTargetPosition=[0.3,0,0.2])


        # load plane
        #plane = p.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])
        self._load_plane()

        # load robot
        
        #self._reset_robot()
        #self._reset_robot_arm()
        # load goal
        #self._load_golf_hole()
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_golf_hole.urdf')
        self.golf_hole = p.loadURDF(fullpath,self.goal_default_pos,[0,0,0,1], useFixedBase=True)
        
        # load ball
        #fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_ball.urdf')
        #self.ball = p.loadURDF(fullpath,[1,0,.5], [0,0,0,1],useFixedBase=True)
        self._load_golf_ball()

        fullpath = os.path.join(os.path.dirname(__file__), "urdf/xarm7.urdf")
        self.xarm = p.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)

        # env setup
        self.action_space = spaces.Box(-1., 1., shape=(3,), dtype='float32')
        self.goal = self._sample_goal(False)



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
        reward = self.compute_reward(obs['achieved_goal'],self.goal)
        done = (reward == 0)
        time.sleep(self.time_step)
        return obs, reward, done, #info
        
    def reset(self):
        #return obs
        self._reset_sim()
        #self.goal = self._sample_goal()
        self.goal = self.goal_default_pos

        return self._get_obs()

    def compute_reward(self, achieved_goal, goal):
        distance = np.linalg.norm(achieved_goal- goal, axis=-1)

        return (distance < self.distance_threshold).astype(np.float32) - 1.

    #  RobotEnv methods
    # ------------------

    def _set_action(self,action):

        assert action.shape == (3,), 'action shape error'
        current_position = np.array(p.getLinkState(self.xarm,self.arm_eef_index)[0])
        new_position = current_position + action * self.max_vel * self.dt
        new_position = np.clip(new_position, self.pos_space.low,self.pos_space.high)
        joint_poses = p.calculateInverseKinematics(self.xarm,self.arm_eef_index,new_position,[1,0,0,0])
        for i in range(1,self.arm_eef_index):
            p.setJointMotorControl2(self.xarm,i,p.POSITION_CONTROL, joint_poses[i-1], force=5*240.)
        
    def move_joint(self,joint_index):
        #for _ in range(60): 
        jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index,[0.55,0,0.2], [1,0,0,0], maxNumIterations = 20)
        for i in range(1, self.arm_eef_index):
            p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1], force = 200) # max=1200
        p.stepSimulation()


    def _get_obs(self):
            #find a way to introduce orientation of gripper as obs

        # robot state
        robot_state = p.getJointStates(self.xarm, np.arange(0,self.num_joints))

        # gripper state
        #gripper_pos = np.array([robot_state[self.gripper_driver_index][0]])
        #gripper_vel = np.array([robot_state[self.gripper_driver_index][1]])
        grip_state = p.getLinkState(self.xarm, self.gripper_base_index, computeLinkVelocity=1)
        grip_pos = np.array(grip_state[0])
        grip_vel = np.array(grip_state[6])

        #golf ball state
        golf_ball_pos, _ = p.getBasePositionAndOrientation(self.golf_ball)

        # distance
        distance = np.array([np.linalg.norm(self.goal-golf_ball_pos,axis=-1)])

        # observation
        obs = np.concatenate((
                    grip_pos, grip_vel, distance
        ),axis= -1)
        #obs = np.concatenate((obs,distance),axis =-1)

        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(golf_ball_pos),
            'desired_goal': self.goal.copy()
        }

       
    def _reset_sim(self):

        self._reset_robot_arm()
        self._reset_golf_ball()
        
    
    def _load_golf_ball(self,coordinates =[0.45,0,0.02]):
            # implement bool for picking default position or random position,
            # add in __init__() self.golf_ball_space = [x,y,z] do [o,p,q]
        ball_shape = p.createVisualShape(shapeType= p.GEOM_SPHERE, radius= 0.02, rgbaColor = [0, 0 ,0.8, 1])
        ball_colision = p.createCollisionShape(shapeType = p.GEOM_SPHERE, radius = 0.02)

        self.golf_ball = p.createMultiBody(baseMass = 0.1,
                                    baseInertialFramePosition = [0,0,0],
                                    baseCollisionShapeIndex = ball_colision,
                                    baseVisualShapeIndex = ball_shape,
                                    basePosition = coordinates,
                                    useMaximalCoordinates = True)

    def _load_plane(self):
        plane = p.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])
        p.changeDynamics(plane,-1, 
                 lateralFriction = .05,
                 rollingFriction = .1,
                 restitution = .7)
        
    def _load_golf_hole(self):
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_golf_hole.urdf')
        self.golf_hole = p.loadURDF(fullpath,self.goal_default_pos,[0,0,0,1], useFixedBase=True)
        print("aa")
    # def _reset_robot(self):
    #     for i in range(17):
    #         p.resetJointState(self.xarm,i,targetValue = self.joint_init_pos[i], targetVelocity = 0)

    def _reset_robot_arm(self):
        for _ in range(60): 
            jointPoses = p.calculateInverseKinematics(self.xarm, self.arm_eef_index, self.gripper_base_default_pos, [1,0,0,0], maxNumIterations = 20)
            for i in range(1, self.arm_eef_index):
                p.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
            p.setJointMotorControl2(self.xarm, 10, p.POSITION_CONTROL, 1, force=1000)
            p.setJointMotorControl2(self.xarm, 11, p.POSITION_CONTROL, 1, force=1000)
            p.setJointMotorControl2(self.xarm, 13, p.POSITION_CONTROL, 1, force=1000)
            p.setJointMotorControl2(self.xarm, 14, p.POSITION_CONTROL, 1, force=1000)
            p.stepSimulation()
    
    def _reset_golf_ball(self):
        p.resetBasePositionAndOrientation(self.golf_ball, self.golf_ball_default_pos, self.start_orientation)
    
        
    def setup_golf_course(self):
        
        raise NotImplementedError
    

    #  Goal methods
    # --------------
    def _sample_goal(self,rand_pos):
        if rand_pos == False:
            goal = self.goal_default_pos
        else:
            goal = np.array(self.goal_space.sample())
        p.resetBasePositionAndOrientation(self.golf_hole, goal, self.startOrientation)
        return goal.copy()


    def make_goal(self,point_position):
        p.addUserDebugLine(point_position, [point_position[0], point_position[1], point_position[2] + 0.01], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0], point_position[1], point_position[2] - 0.01], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] - 0.01, point_position[1], point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] + 0.01, point_position[1], point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0], point_position[1] + 0.01, point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] , point_position[1]- 0.01, point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)