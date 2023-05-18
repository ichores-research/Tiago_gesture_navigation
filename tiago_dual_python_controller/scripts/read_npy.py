import numpy as np

img_array = np.load('/home/guest/MotionBERT-main/demo/res/X3D.npy')

from matplotlib import pyplot as plt

#plt.imshow(img_array, cmap='gray')
plt.show()

print(len(img_array[0]))
print(img_array)