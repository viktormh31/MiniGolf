import os
import gymnasium as gym
import gymnasium_robotics
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import numpy as np
import time
import random
from memory import HerBuffer
#import matplotlib.pyplot as plt
from tqdm import tqdm, trange
