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
hole = p.loadURDF(fullpath,[1,0,-.09], [0,0,0,1],useFixedBase=True)


for i in range(1000):
    p.stepSimulation()
    time.sleep(.01)
    print(i)
    print(p.getBasePositionAndOrientation(golf_ball))
    if i == 500:
        p.resetBasePositionAndOrientation(golf_ball, [0,0,0.1], [0,0,0,1])


time.sleep(20)
