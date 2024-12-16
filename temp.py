import numpy as np
import torch
import os
from XarmGolfEnv import XarmRobotEnv
from XarmReach import XarmRobotReach
#from SAC_with_temperature_v2 import Agent
#import SAC_with_temperature_v2
import SAC_v3
import time
config = {

    'GUI' : True,
    'reward_type' : "sparse",
}


#env =XarmRobotReach(config)


import pandas as pd

x = [1, 2, 3, 4, 5]
y = [2, 4, 1, 3, 7]

df = pd.DataFrame({'X': x, 'Y': y})
df.plot(x='X', y='Y', kind='line', marker='o')



print(np.arange(5, 5 + 4) % 6)




print("a")