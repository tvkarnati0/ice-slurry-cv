import matplotlib.pyplot as plt
import cv2

img = cv2.imread("images/ice_slurry1.png")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # matplotlib expects RGB, not BGR

plt.imshow(img_rgb)
plt.show()