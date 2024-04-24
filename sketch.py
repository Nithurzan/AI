
import cv2

img = cv2.imread("caption.jpeg")
grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
invertImg = cv2.bitwise_not(grayImg)
blurImg = cv2.GaussianBlur(invertImg, (21,21),0)
invetBlurImg = cv2.bitwise_not(blurImg)
sketch = cv2.divide(grayImg,invetBlurImg, scale=256.0)

cv2.imwrite("sketch.png", sketch)