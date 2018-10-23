import cv2
from PIL import Image
import numpy as np
mask = np.zeros([563, 904], np.int32)
for i in range(280, 400):
    for j in range(400, 500):
        mask[i][j] = 255

img = Image.fromarray(mask.astype(np.uint8))
img.save('/home/hzj/ImageImpainting/NS/mask.jpg')

img = cv2.imread("/home/hzj/ImageImpainting/target.png")
mask = cv2.imread('/home/hzj/ImageImpainting/NS/mask.jpg', 0)
dst = cv2.inpaint(img, mask, 3, cv2.INPAINT_NS)
cv2.imwrite('/home/hzj/ImageImpainting/NS/dst.png', dst)