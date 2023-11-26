# 李佳朗
# 开发时间：2023
import cv2
import numpy as np
cap = cv2.VideoCapture(0)# 打开摄像头
myColors = [[90,48,0,118,255,255]]# 定义颜色范围
def findColor(img, myColors):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)# 将图像转换为HSV颜色空间
    masks = []# 存储掩码
    contours_list = []  # 存储每个颜色对应的所有轮廓
    for color in myColors:
        lower = np.array(color[0:3])# 颜色范围的下限
        upper = np.array(color[3:6]) # 颜色范围的上限
        mask = cv2.inRange(imgHSV, lower, upper)# 创建颜色掩膜
        masks.append(mask)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # 查找轮廓
        contours_list.append(contours)
        cv2.imshow('11',mask)# 显示掩膜图像
    return masks, contours_list
def describe(contours_list):
    for contours in contours_list:
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 500: # 根据面积筛选轮廓
                x, y, w, h = cv2.boundingRect(cnt)# 获取轮廓的边界框
                cv2.rectangle(imgResult, (x, y), (x + w, y + h), (0, 255, 0), 2) # 绘制边界框
                location_x = (2 * x + w) // 2
                location_y = (2 * y + h) // 2
                cv2.putText(imgResult, "blue", (location_x - 100, location_y), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 0, 255), 1, cv2.LINE_AA) # 绘制文字标签
while True:
    success, img = cap.read() # 读取摄像头图像
    imgResult = img.copy()# 复制原始图像
    masks, contours_list = findColor(img, myColors)
    describe(contours_list)
    cv2.imshow("imgcolor", imgResult)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break