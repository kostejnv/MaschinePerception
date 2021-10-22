#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import cv2
from matplotlib import pyplot as plt


# ### A) show image

# In[10]:


I = cv2.imread('images/umbrellas.jpg')
I = cv2.cvtColor(I, cv2.COLOR_BGR2RGB)
plt.imshow(I)
plt.show()


# ### B) convert to grayscale

# In[11]:


# i_gray = cv2.cvtColor(I, code=cv2.COLOR_BGR2GRAY)
float_I = I.astype(float)
grayscale_I = (float_I[:, :, 0] + float_I[:, :, 1] + float_I[:, :, 2]) / 3
plt.imshow(grayscale_I, cmap='gray')
plt.show()


# ### C) show only part of picture

# In[12]:


cutout=I[130:260, 240:450, 1]
plt.imshow(cutout, cmap='gray')
plt.show()


# Question: Why would you use different color maps?
# 
# The pyplot library is not only for pictures but also for other graphs. There can be useful to show results in other color map. Also, we can show only red part of picture, then it will be also better use different colormap.

# ### D) inverting part of image

# In[13]:


def invert_image_part(src, start, end):
    src_copy = np.copy(src)
    for y in range(start[0], end[0]+1):
        for x in range(start[1], end[1]+1):
            for channel in range(3):
                src_copy[y,x,channel] = 255 - src[y,x,channel]
    return src_copy

inverted_I = invert_image_part(I,(120,240), (260,450))

plt.subplot(1,2,1)
plt.imshow(I)
plt.subplot(1,2,2)
plt.imshow(inverted_I)
plt.show()


# Question: How is inverting a grayscale value defined for uint8?
# 
# Don't understand the question. I think it is also 255 - ((r+g+b)/3)

# ### E) reducing grayscale level

# In[14]:


gray_I = cv2.cvtColor(I, code=cv2.COLOR_BGR2GRAY)
gray_I = gray_I.astype(float)
gray_I *= 63/255.0
gray_I = gray_I.astype('uint8')

plt.subplot(1,2,1)
plt.imshow(grayscale_I, cmap='gray')
plt.subplot(1,2,2)
plt.imshow(gray_I, cmap='gray', vmax=255)
plt.show()

