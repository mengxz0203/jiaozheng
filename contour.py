import cv2


def _caculate_image_contour(img):
    h, w = img.shape[:2]  # 获取图像的高和宽
    print("h:", h)
    print("w:", w)
    canny = cv2.Canny(img, 200, 2.5)  # 使用canny算法得到轮廓图
    cv2.imshow("canny", canny)
    cv2.waitKey()
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    closed = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)  # 对轮廓图执行一次闭运算，加强轮廓的连通性
    _, contours, _ = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 从轮廓图中提取出所有轮廓的数据
    # 对每一个轮廓计算与其外切的最小矩形的坐标和面积，并根据面积大小进行排序
    size_dict = {}
    for contour in contours:
        x1, y1, cnt_w, cnt_h = cv2.boundingRect(contour)  # 得到该轮廓外接矩形的左上角坐标以及宽高
        x2 = x1 + cnt_w
        y2 = y1 + cnt_h
        size = cnt_w * cnt_h
        size_dict[((x1, y1), (x2, y2))] = size  # 键为由矩形左上角和右下角坐标组成的一个二元组，值为矩形面积
    size_list = sorted(size_dict.items(), key=lambda x: x[1], reverse=True)  # 根据矩形面积进行排序
    expected_item_index = 0
    expected_item = size_list[expected_item_index]
    while expected_item[1] > h * w * 0.95:  # 取面积最大的矩形作为轮廓，但如果该矩形面积与原图像面积接近，则取第二大面积的矩形作为轮廓
        expected_item_index += 1
        expected_item = size_list[expected_item_index]
    expected_x1, expected_y1 = expected_item[0][0]  # 选取的矩形左上角坐标
    expected_x2, expected_y2 = expected_item[0][1]  # 选取的矩形右下角坐标
    contour_map = [int(expected_x1), int(expected_y1), int(expected_x2), int(expected_y2)]
    return contour_map


def calculating_image_contour(img):  # 获取设备轮廓
    contour_map = _caculate_image_contour(img)  # 初步截图设备（带边框）轮廓
    x_left, y_top, x_right, y_down = contour_map
    cv2.rectangle(img, (x_left, y_top), (x_right, y_down), (0, 0, 255), 5)
    # img_contour = img[y_top:y_down,x_left:x_right]
    cv2.imshow("contour", img)
    cv2.waitKey()


if __name__ == "__main__":
    img = cv2.imread("1.png")
    shrinkedPic = cv2.pyrDown(img)
    shrinkedPic = cv2.pyrDown(shrinkedPic)
    cv2.imshow("shrinked", shrinkedPic)
    cv2.waitKey(0)

    grayPic = cv2.cvtColor(shrinkedPic, cv2.COLOR_BGR2GRAY)
    cv2.imshow("gray", grayPic)
    cv2.waitKey(0)

    # 中值滤波降噪
    median = cv2.medianBlur(grayPic, 7)
    cv2.imshow("medianBlur", median)
    cv2.waitKey()

    # 二值化
    ret, binary = cv2.threshold(median, 170, 255, cv2.THRESH_BINARY)
    cv2.imshow("binary", binary)
    cv2.waitKey()

    calculating_image_contour(binary)
