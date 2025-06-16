import numpy as np
import matplotlib.pyplot as plt
from utils import *

# Discretization parameters
dx = 0.01 # space step
t_end = 5 # end of simulation time 
gamma = 1.4
c = 0.8

# Grid parameters
x_start = 0
x_end = 10+dx

delta_x_ = 1 # Interval of the "disturbance"
