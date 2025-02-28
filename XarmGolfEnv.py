import numpy as np
import pybullet as p
import pybullet_data
import os
from gymnasium import spaces
import pybullet_utils.bullet_client as bc
import sys

class XarmRobotGolf():

    def __init__(self, config):
        self.time_step = 1./240
        
        # env params
        self.startOrientation = p.getQuaternionFromEuler([0,0,0])
        self.pos_space = spaces.Box(low=np.array([0.35,-0.4,0.02],dtype=np.float32),
                                    high=np.array([0.8,0.4,0.02],dtype=np.float32),)
        self.dt = self.time_step * 40
        self.action_space = spaces.Box(-1., 1., shape=(4,), dtype='float32')
        self.phase = 1
        self.difficulty = 0.9

        # bullet setup
        
        if config['GUI']:
            self.physics_client = bc.BulletClient(p.GUI)
            self.physics_client.resetDebugVisualizerCamera(cameraDistance=2, cameraYaw=0, cameraPitch=-10, cameraTargetPosition=[0.5,0,0.2])
           
        else:
            self.physics_client = bc.BulletClient(p.DIRECT)
        
        self.physics_client.resetSimulation()
        self.physics_client.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.physics_client.setGravity(0.0,0.0,-9.81)
        self.physics_client.setRealTimeSimulation(0)
        self.physics_client.resetDebugVisualizerCamera(cameraDistance=0.7, cameraYaw=90, cameraPitch=-20, cameraTargetPosition=[1,0,0.4])


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
        self.start_orientation = self.physics_client.getQuaternionFromEuler([0,0,0])
        self.gripper_base_default_pos = [0.35, 0., 0.02]
        self.max_vel = 1
        self.max_angle = np.radians(45)
        # load robot
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7.urdf')
        self.xarm = self.physics_client.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)


        # hole params
        self.hole_space = spaces.Box(low=np.array([0.9, -0.2, 0.02],dtype=np.float32)
                                    ,high=np.array([1.1 ,0.2, 0.02],dtype=np.float32))
        self.hole_default_pos = np.array([0.9,0.05,0.02])
        self.distance_threshold = 0.05

        # load goal
        
        self._load_golf_hole()
        

        # ball params
        self.golf_ball_space = spaces.Box(low=np.array([0.45, -0.1, 0.02],dtype=np.float32)
                                        ,high=np.array([0.6 ,0.1, 0.02],dtype=np.float32))
        self.golf_ball_default_pos = [0.45,0,0.02]
        self.golf_ball_pos = self.golf_ball_default_pos
        
        # load ball
        self._load_golf_ball()


      
     
    #  Basic methods
    # ---------------

    def step(self,action):

        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        for i in range(5):
            self.physics_client.stepSimulation() # stavi vise stepova
        obs = self._get_obs()
        reward = self.compute_reward(obs['achieved_goal'],obs['desired_goal'])
        done = reward + 1
        #time.sleep(self.time_step)
        return obs, reward, done, #info
        
    def reset(self):
        #return obs
        self._reset_sim()
        self._reset_golf_hole()
        if self.phase == 1:
            #self._sample_goal()
            #self._reset_golf_hole()
            self._reset_golf_ball()
        elif self.phase == 2:
            self._sample_golf_ball()
            #self._reset_golf_hole()
        elif self.phase == 3:
            self._sample_golf_ball()
            self._sample_goal()
        
        return self._get_obs()

    def compute_reward(self, achieved_goal, goal):
        distance = np.linalg.norm(achieved_goal - goal, axis=-1)

        return (distance < self.distance_threshold).astype(np.float32) - 1.

    #  RobotEnv methods
    # ------------------

    # current_gripper_base_value = np.array(p.getJointState(self.xarm, self.gripper_base_index)[0])
    #     new_gripper_base_value = current_gripper_base_value = action[4] * self.max_vel *self.dt
    #     new_gripper_base_value = np.clip(new_gripper_base_value,)


    def _set_action(self,action):
        # action = xyz, value for gripper joint
        #assert action.shape == (3,), 'action shape error'
        assert action.shape == (4,), 'action shape error'
        current_position = np.array(self.physics_client.getLinkState(self.xarm,self.tcp_index)[0])
        new_position = current_position + action[:3] * self.max_vel * self.dt
        new_position = np.clip(new_position, self.pos_space.low,self.pos_space.high)
        new_quaternion = self._calculate_quaternion(action[3])
        
        joint_poses = self.physics_client.calculateInverseKinematics(self.xarm,self.tcp_index,new_position,new_quaternion)
        for i in range(1,self.arm_eef_index):
            self.physics_client.setJointMotorControl2(self.xarm,i,p.POSITION_CONTROL, joint_poses[i-1])
 

    def _calculate_quaternion(self,action):
        current_quaternion = self.physics_client.getLinkState(self.xarm,self.tcp_index)[1]
        current_euler = np.array(self.physics_client.getEulerFromQuaternion(current_quaternion))
        scaled_turn_action = action * self.dt
        new_euler_2 = current_euler[2] + scaled_turn_action
        new_euler_2 = np.clip(new_euler_2,-self.max_angle,self.max_angle)
        new_euler = np.array([current_euler[0],current_euler[1],new_euler_2])
        new_quaternion = self.physics_client.getQuaternionFromEuler(new_euler)
        return new_quaternion

    def _get_obs(self):
            #find a way to introduce orientation of gripper as obs

        tcp_state = self.physics_client.getLinkState(self.xarm, self.tcp_index, computeLinkVelocity=1)
        tcp_pos = np.array(tcp_state[0])
        tcp_rot = self.physics_client.getEulerFromQuaternion(np.array(tcp_state[1]))
        tcp_vel = np.array(tcp_state[6])

        #golf ball state
        self.golf_ball_pos, ball_orientation = self.physics_client.getBasePositionAndOrientation(self.golf_ball)
        ball_orientation = self.physics_client.getEulerFromQuaternion(ball_orientation)
        ball_vel = self.physics_client.getBaseVelocity(self.golf_ball)[0]
        relative_ball_vel = ball_vel - tcp_vel
        #ball_ang_vel = p.getBaseVelocity(self.golf_ball)[1]

        
        # distance
        #distance = np.array([np.linalg.norm(self.hole_pos-self.golf_ball_pos,axis=-1)])
        distance = self.golf_ball_pos - tcp_pos
        
        # observation
        obs = np.concatenate((
                    tcp_pos,tcp_rot, tcp_vel, self.golf_ball_pos, relative_ball_vel,ball_orientation, distance
        ),axis= -1)
        #obs = np.concatenate((obs,distance),axis =-1)

        return {
            'observation': obs.copy(),
            'achieved_goal': np.squeeze(self.golf_ball_pos),
            'desired_goal': self.hole_pos.copy()
        }

       
    def _reset_sim(self):

        self.reset_robot()
        #self._reset_golf_ball()
        
    
    def _load_golf_ball(self):
            # implement bool for picking default position or random position,
            # add in __init__() self.golf_ball_space = [x,y,z] do [o,p,q]
        ball_shape = self.physics_client.createVisualShape(shapeType= p.GEOM_SPHERE, radius= 0.02, rgbaColor = [0, 0 ,0.8, 1])
        ball_colision = self.physics_client.createCollisionShape(shapeType = p.GEOM_SPHERE, radius = 0.02)

        self.golf_ball = self.physics_client.createMultiBody(baseMass = 0.1,
                                    baseInertialFramePosition = [0,0,0],
                                    baseCollisionShapeIndex = ball_colision,
                                    baseVisualShapeIndex = ball_shape,
                                    basePosition = self.golf_ball_default_pos,
                                    useMaximalCoordinates = True)

    def _load_plane(self):
        plane = self.physics_client.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])
        self.physics_client.changeDynamics(plane,-1, 
                 lateralFriction = .05,
                 rollingFriction = .1,
                 restitution = .7)
        
    def _load_golf_hole(self):
        fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_golf_hole.urdf')
        self.hole = self.physics_client.loadURDF(fullpath,self.hole_default_pos,self.start_orientation, useFixedBase=True)
        self.hole_pos = self.hole_default_pos

    def _reset_robot_arm(self):
        for _ in range(200): 
            jointPoses = self.physics_client.calculateInverseKinematics(self.xarm, self.tcp_index, self.gripper_base_default_pos, [1,0,0,0], maxNumIterations = 20)
            #jointPoses[10,11,13,14] = 1
            #p.setJointMotorControlArray(self.xarm,np.arange(1,self.arm_eef_index), p.POSITION_CONTROL, jointPoses[i-1])
            joint_indexes = np.arange(0,self.arm_eef_index)
           #p.resetJointState(self.xarm,joint_indexes,jointPoses)

            for i in range(1, self.arm_eef_index):
                self.physics_client.setJointMotorControl2(self.xarm, i, p.POSITION_CONTROL, jointPoses[i-1]) # max=1200
            self.physics_client.setJointMotorControl2(self.xarm, 10, p.POSITION_CONTROL, 1, force=1000)
            self.physics_client.setJointMotorControl2(self.xarm, 11, p.POSITION_CONTROL, 1, force=1000)
            self.physics_client.setJointMotorControl2(self.xarm, 13, p.POSITION_CONTROL, 1, force=1000)
            self.physics_client.setJointMotorControl2(self.xarm, 14, p.POSITION_CONTROL, 1, force=1000)
            self.physics_client.stepSimulation()
        states = []
        for i in range(1, self.arm_eef_index):
            states.append(self.physics_client.getJointState(self.xarm,i)[0])
       
    
    def reset_robot(self):
        self.joint_init_pos = [0, 0.27050686544800806, -0.005340887396375177, -0.2711492861468919
                               , 0.4600086544592818, -0.003261038126179985, 0.46525864138699663
                               , 0.0022592575465344170,0,0,0.85,0.85,0.85,0.85,0.85,0.85]
        
        for i in range(16):
            self.physics_client.resetJointState(self.xarm,i,targetValue = self.joint_init_pos[i], targetVelocity = 0)


    def _reset_golf_ball(self):

        self.physics_client.resetBasePositionAndOrientation(self.golf_ball, self.golf_ball_default_pos, self.start_orientation)
    
    def _sample_golf_ball(self):
        
        self.golf_ball_pos = np.array(self.golf_ball_space.sample())
        self.physics_client.resetBasePositionAndOrientation(self.golf_ball, self.golf_ball_pos, self.start_orientation)

    def _reset_golf_hole(self):
        self.physics_client.resetBasePositionAndOrientation(self.hole,self.hole_default_pos,self.start_orientation)
        
    def setup_golf_course(self):
        raise NotImplementedError
    

    #  Goal methods
    # --------------
    def _sample_goal(self):

        self.hole_pos = np.array(self.hole_space.sample())
        #self.hole_pos[0]= self.difficulty
        self.physics_client.resetBasePositionAndOrientation(self.hole, self.hole_pos, self.startOrientation)


    def make_goal(self,point_position):

    
        p.addUserDebugLine(point_position, [point_position[0], point_position[1], point_position[2] + 0.01], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0], point_position[1], point_position[2] - 0.01], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] - 0.01, point_position[1], point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] + 0.01, point_position[1], point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0], point_position[1] + 0.01, point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)
        p.addUserDebugLine(point_position, [point_position[0] , point_position[1]- 0.01, point_position[2]], lineColorRGB=[1, 0, 0], lineWidth=5)


    def close(self):
        self.physics_client.disconnect()

    def test_reset(self):
        p.resetSimulation()
