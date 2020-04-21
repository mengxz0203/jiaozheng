import cv2
import numpy as np

# 读取名称为 p19.jpg的图片
img = cv2.imread("IMG_4238.JPG", 1)
img_org = cv2.imread("IMG_4238.JPG", 1)

# 得到图片的高和宽
img_height, img_width = img.shape[:2]

# 定义对应的点
points1 = np.float32([[75, 55], [340, 55], [33, 435], [400, 433]])
points2 = np.float32([[0, 0], [360, 0], [0, 420], [360, 420]])

# 计算得到转换矩阵
M = cv2.getPerspectiveTransform(points1, points2)

# 实现透视变换转换
processed = cv2.warpPerspective(img, M, (360, 420))

# 显示原图和处理后的图像
cv2.imshow("org", img_org)
cv2.imshow("processed", processed)

cv2.waitKey(0)
