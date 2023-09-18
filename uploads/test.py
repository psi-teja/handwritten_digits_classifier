# Implementation of matplotlib function
import numpy as np
import matplotlib.cbook as cbook
import matplotlib.image as image
import matplotlib.pyplot as plt
  
with open('example.jpeg', "r") as image_file:
    image = plt.imread(image_file)
  
fig, ax = plt.subplots()
ax.imshow(image)
ax.axis('off')
  
plt.title('matplotlib.pyplot.imread() function Example', 
                                     fontweight ="bold")
plt.show()
