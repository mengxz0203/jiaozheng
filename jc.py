import cv2
import numpy as np

# 读入图片
# img = cv2.imread('IMG_2961.JPG')
# img = cv2.imread('IMG_4218.JPG')
# img = cv2.imread('IMG_4238.JPG')
# img = cv2.imread('IMG_4380.JPG')
# img = cv2.imread('a.jpg')
# img = cv2.imread('b.jpg')
img = cv2.imread('c.jpg')
# img = cv2.imread('pic.png')
# cv2.imshow("img", img)


# 缩小尺寸
shrinkedPic = cv2.pyrDown(img)
shrinkedPic = cv2.pyrDown(shrinkedPic)
# cv2.imshow("img", shrinkedPic)
# cv2.waitKey()

# 灰度化
gray = cv2.cvtColor(shrinkedPic, cv2.COLOR_BGR2GRAY)
# cv2.imshow("gray", gray)
# cv2.waitKey()

# 中值滤波降噪
# median = cv2.medianBlur(gray, 7)
# cv2.imshow("medianBlur", median)
# cv2.waitKey()
blur = cv2.GaussianBlur(gray, (5, 5), 0)
cv2.imshow("medianBlur", blur)
cv2.waitKey()


# 二值化
# ret, binary = cv2.threshold(median, 40, 255, cv2.THRESH_BINARY)
binary = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
# ret, binary = cv2.threshold(median, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# print("ret", ret)
cv2.imshow("binary", binary)
cv2.waitKey()

kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

erode = cv2.erode(binary, kernel)
cv2.imshow("erode", erode)
cv2.waitKey()

kernel_1 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)  # 对轮廓图执行一次闭运算，加强轮廓的连通性
dst = cv2.dilate(erode, kernel_1)
cv2.imshow("dst", dst)
cv2.waitKey()

# 边缘检测
canny = cv2.Canny(dst, 50, 150)
cv2.imshow("canny", canny)
cv2.waitKey()

kernel_2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel_2)
cv2.imshow("closed", closed)
cv2.waitKey(0)

find, contours, hierarchy = cv2.findContours(closed, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)  # 从轮廓图中提取出所有轮廓的数据
# cv2.imshow("find", find)
# cv2.waitKey(0)

h, w = dst.shape[:2]
print("h:", h)
print("w:", w)
linePic = np.zeros((h, w, 3))

cv2.drawContours(linePic, contours, -1, (0, 0, 255), 1)
cv2.imshow("line", linePic)
cv2.waitKey()

cv2.drawContours(shrinkedPic, contours, -1, (0, 0, 255), 1)
cv2.imshow("line", shrinkedPic)
cv2.waitKey()

maxArea = 0
length = len(contours)
for index in range(length):
    if cv2.contourArea(contours[index]) > cv2.contourArea(contours[maxArea]):
        maxArea = index

polyContours = cv2.approxPolyDP(contours[maxArea], 10, True)

hull = []
length = len(polyContours)
for index in range(length):
    hull = cv2.convexHull(polyContours[index])
    print(hull)

polyPic = np.zeros((h, w, 3))
# cv2.drawContours(polyPic, polyContours, -1, (0, 255, 0), 1)
cv2.polylines(polyPic, [polyContours], True, (0, 0, 255), 2)
cv2.imshow("poly", polyPic)
cv2.waitKey()

cv2.polylines(shrinkedPic, [polyContours], True, (0, 255, 0), 2)
cv2.imshow("poly", shrinkedPic)
cv2.waitKey()

# cv2.waitKey(0)
cv2.destroyAllWindows()
