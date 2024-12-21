import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import gymnasium as gym
from gymnasium import spaces






p.connect(p.GUI)
p.resetDebugVisualizerCamera( cameraDistance=1.5, cameraYaw=0, cameraPitch=-45, cameraTargetPosition=[0.3,0.1,-0.1])

p.resetSimulation()
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0.0,0.0,-9.81)
p.setRealTimeSimulation(0)
plane = p.loadURDF("plane.urdf", [0,0,0], [0,0,0,1])

#fullpath = os.path.join(os.path.dirname(__file__), "urdf/xarm7.urdf")
#xarm = p.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)

fullpath = os.path.join(os.path.dirname(__file__), 'urdf/xarm7.urdf')
xarm = p.loadURDF(fullpath, [0,0,0], [0,0,0,1], useFixedBase = True)


cur_pos = p.getLinkState(xarm,16)[0]


"""
fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_ball.urdf')
sphere = p.loadURDF(fullpath,[0.5,0,0.6],useFixedBase=True)

ball_shape = p.createVisualShape(shapeType= p.GEOM_SPHERE, radius= 0.05, rgbaColor = [0, 0 ,0.8, 1])
ball_colision = p.createCollisionShape(shapeType = p.GEOM_SPHERE, radius = 0.05)

golf_ball = p.createMultiBody(baseMass = 0.1,
                              baseInertialFramePosition = [0,0,0],
                              baseCollisionShapeIndex = ball_colision,
                              baseVisualShapeIndex = ball_shape,
                              basePosition = [0,0,2],
                              useMaximalCoordinates = True)

golf_ball2 = p.createMultiBody(baseMass = 0.1,
                              baseInertialFramePosition = [0,0,0],
                              baseCollisionShapeIndex = ball_colision,
                              baseVisualShapeIndex = ball_shape,
                              basePosition = [0.3,0.01,.05],
                              useMaximalCoordinates = False)

p.changeDynamics(plane,-1, 
                 lateralFriction = .2,
                 rollingFriction = .1,
                 restitution = .7)
p.changeDynamics(golf_ball2,-1,
                restitution = .3)
p.changeDynamics(golf_ball,-1,
                restitution = .3)

fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_golf_hole.urdf')
hole = p.loadURDF(fullpath,[1,0,0], [0,0,0,1],useFixedBase=True)
"""   


max_angle = np.radians(90)
for i in range(1000):
    p.stepSimulation()
    time.sleep(.01)
    orientation = p.getLinkState(xarm,16)[1]
    cur_eul = np.array(p.getEulerFromQuaternion(orientation))
    cur_eul[2] = cur_eul[2] - 0.05
    cur_eul[2] = np.clip(cur_eul[2],-max_angle,max_angle)
    new_eul = np.array([cur_eul[0],cur_eul[1],cur_eul[2]])
   
    new_quat = p.getQuaternionFromEuler(new_eul)
    joint_poses = p.calculateInverseKinematics(xarm,16,cur_pos,new_quat)
    for i in range(1,8):
        p.setJointMotorControl2(xarm,i,p.POSITION_CONTROL, joint_poses[i-1])


time.sleep(20)
