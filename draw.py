#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:40:53 2020

@author: teja
"""

import cv2
import numpy as np

# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(img,(x,y),10,(255,0,0),-1)

# Create a black image, a window and bind the function to window
img = np.ones((512,512,1), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)

while(1):
    cv2.imshow('image',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cv2.destroyAllWindows()