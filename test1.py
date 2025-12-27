import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import gymnasium as gym
from gymnasium import spaces
import cv2 as cv
import torch
from ultralytics import YOLO



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




fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_ball.urdf')
sphere = p.loadURDF(fullpath,[0.5,0,0.6],useFixedBase=True)

ball_shape = p.createVisualShape(shapeType= p.GEOM_SPHERE, radius= 0.02, rgbaColor = [0, 0 ,0.8, 1])
ball_colision = p.createCollisionShape(shapeType = p.GEOM_SPHERE, radius = 0.02)

golf_ball = p.createMultiBody(baseMass = 0.1,
                              baseInertialFramePosition = [0,0,0],
                              baseCollisionShapeIndex = ball_colision,
                              baseVisualShapeIndex = ball_shape,
                              basePosition = [0.45,0.1,0.02],
                              useMaximalCoordinates = True)

golf_ball2 = p.createMultiBody(baseMass = 0.1,
                              baseInertialFramePosition = [0,0,0],
                              baseCollisionShapeIndex = ball_colision,
                              baseVisualShapeIndex = ball_shape,
                              basePosition = [1.22,0.01,.05],
                              useMaximalCoordinates = False)
"""
p.changeDynamics(plane,-1, 
                 lateralFriction = .2,
                 rollingFriction = .1,
                 restitution = .7)
p.changeDynamics(golf_ball2,-1,
                restitution = .3)
p.changeDynamics(golf_ball,-1,
                restitution = .3)
"""
fullpath = os.path.join(os.path.dirname(__file__), 'urdf/my_golf_hole.urdf')
hole = p.loadURDF(fullpath,[1.1,0.2,0], [0,0,0,1],useFixedBase=True)
   
cam_width, cam_height = 1280, 1280
fov = 60  # Field of view
aspect = cam_width / cam_height
near = 0.1
far = 10

cam_eye = [0.8, 0, .8]  # Camera position
cam_target = [0.79,0, 0]  # Where the camera is looking
cam_up = [0., 0., 1]  # "Up" direction
view_matrix = p.computeViewMatrix(cam_eye, cam_target, cam_up)
proj_matrix = p.computeProjectionMatrixFOV(fov, aspect, near, far)
#model = YOLO('models/yolo_model_v2.pt')
model = YOLO('models/golf_recognition_yolov8.pt')

def get_coords_from_yolo(yolo_coords):
        y1,x1,y2,x2 = yolo_coords
        real_x1 = (x1 + x2)/(2*1280) + 0.27
        real_y1 = (y1 + y2)/(2*1280) -0.5

        breal_x1 = (x1 + x2)*0.000679687/2 + 0.35
        breal_y1 = (y1 + y2)*0.000679687/2 -0.435
        #breal_x1 = (x1 + x2)*0.000849609/2 + 0.35
        #breal_y1 = (y1 + y2)*0.000849609/2 -0.435

        return torch.tensor([real_x1.round(decimals=4),real_y1.round(decimals=4)])
    

def process_image(img,time_step):
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
        results = model(rgb_image, verbose = False)
        end = time.time()
        processing_time = start - end
        #print(1000*(end - start))
        img = results[0].plot()
        cv.imshow("Live Tracking",rgb_image)
        cv.waitKey(1)
        for r in results:
            #print(r.boxes)
            for detected_object in r.boxes.data:
                #print(detected_object)
                label_number = detected_object[-1]
                coords = detected_object[:4]
                real_coords = get_coords_from_yolo(coords)

                if label_number == 0.0:
                    golf_ball_coords = real_coords
                    ball_found = True
                elif label_number == 1.0:
                    golf_hole_coords = real_coords
                    hole_found = True

        #cv.imshow("Live Tracking", rgb_image)
        #golf_ball_speed = calculate_speed(golf_ball_coords, time_step)


        return golf_ball_coords, golf_hole_coords,# golf_ball_speed
    
def calculate_speed(self,current_pos, current_step):
        if self.last_position is None or self.last_time is None:
            speed = torch.tensor([0,0,0], device= self.actor.device)
        else:
            delta_pos = current_pos- self.last_position
            delta_t = current_step - self.last_time
            speed = (delta_pos / delta_t) if delta_t > 0 else torch.tensor([0,0,0]).to(self.actor.device)

        

        return speed



max_angle = np.radians(90)
for i in range(1000):
    _, _, rgb_img, _, _ = p.getCameraImage(cam_width, cam_height, view_matrix, proj_matrix,)
    ball_coords , hole_coords = process_image(rgb_img,1)
  