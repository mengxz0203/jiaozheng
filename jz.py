import cv2
import numpy as np

'''
边框 最小矩形区域和最小闭圆的轮廓
'''
img = cv2.pyrDown(cv2.imread('1.png', cv2.IMREAD_UNCHANGED))

# 转换为灰色gray_img
gray_img = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2GRAY)

# 对图像二值化处理 输入图像必须为单通道8位或32位浮点型
ret, thresh = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

# 寻找最外面的图像轮廓 返回修改后的图像 图像的轮廓  以及它们的层次
image, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

print(type(contours))
print(type(contours[0]))
print(len(contours))

# 遍历每一个轮廓
for c in contours:
    # 找到边界框的坐标
    x, y, w, h = cv2.boundingRect(c)
    # 在img图像上 绘制矩形  线条颜色为green 线宽为2
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 找到最小区域
    rect = cv2.minAreaRect(c)

    # 计算最小矩形的坐标
    box = cv2.boxPoints(rect)

    # 坐标转换为整数
    box = np.int0(box)

    # 绘制轮廓  最小矩形 blue
    cv2.drawContours(img, [box], 0, (255, 0, 0), 3)

    # 计算闭圆中心店和和半径
    (x, y), radius = cv2.minEnclosingCircle(c)

    # 转换为整型
    center = (int(x), int(y))
    radius = int(radius)

    # 绘制闭圆
    img = cv2.circle(img, center, radius, (0, 255, 0), 2)

cv2.drawContours(img, contours, -1, (0, 0, 255), 2)
cv2.imshow('contours', img)
cv2.waitKey()
