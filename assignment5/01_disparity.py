#!/usr/bin/env python
# coding: utf-8

# In[2]:





# In[3]:


# libraries
import numpy as np
import cv2
from matplotlib import pyplot as plt


# # 1) Disparity
# 
# ### A) deriving of disparity
# 
# in the notebook
# 
# ### B) plot disparity

# In[12]:


f = 2.5 #(mm)
T = 120 #(mm)

xs = np.linspace(5, 300, num=1000)
ys = list(map(lambda pz: f*T/pz, xs))
plt.plot(xs, ys)
plt.title("Disparity")
plt.xlabel("depth (mm)")
plt.ylabel("disparity (mm)")
plt.show()


# ### c) compute task
# in the notebook
# 
