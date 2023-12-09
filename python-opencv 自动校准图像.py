import cv2
import numpy as np
def empty(a):
    pass
def stackImages(scale, imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2:
                    imgArray[x][y] = cv2.cvtColor(imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank] * rows
        hor_con = [imageBlank] * rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None, scale, scale)
            if len(imgArray[x].shape) == 2:
                imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver

path = r"C:\Users\ljl20\Pictures\book.jpg"
cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",640,240)
cv2.createTrackbar("Hue Min","TrackBars",51,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",179,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",4,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",255,255,empty)
cv2.createTrackbar("Val Min","TrackBars",212,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)
while True:
    img = cv2.imread(path)
    imgHSV = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max", "TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min", "TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max", "TrackBars")
    v_min = cv2.getTrackbarPos("Val Min", "TrackBars")
    v_max = cv2.getTrackbarPos("Val Max", "TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)
    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imgHSV,lower,upper)
    mask = cv2.medianBlur(mask,5)
    #imgResult = cv2.bitwise_and(img,img,mask=mask)
    imgStack = stackImages(0.6,([img,imgHSV,mask]))
    # 寻找轮廓并计算角点坐标
    cv2.imshow("mask",mask)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    points = []
    min_x = float("inf")
    min_y = float("inf")
    max_x = float("-inf")
    max_y = float("-inf")

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)  # 获取到的一个点
        for point in approx:
            x = point[0][0]
            y = point[0][1]
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)

            cl = cv2.circle(img, (x, y), 5, (0, 250, 0), cv2.FILLED)
            cv2.putText(img, f"({x},{y})", (x - 20, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.rectangle(img, (min_x, min_y), (max_x, max_y), (0, 0, 255), 2)
    cv2.imshow("img", img)

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True) # 获取到的一个点
        # points.append(approx[0])
        for point in approx:
            cl = cv2.circle(img, (point[0][0],point[0][1]), 5, (0, 250, 0), cv2.FILLED)
            cv2.putText(img, f"({point[0][0]},{point[0][1]})", (point[0][0] - 20, point[0][1] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    print(point," ",cv2.rectangle)
    print(point," ",img.shape)
    print(approx)
    [left,x1]=[165,118]
    [high,y1]=[407,46]
    [low,y2]=[380,443]
    [right,x2]=[630,321]
    pts1 = np.float32(points)  # 将角点坐标转换为浮点型数组
    width, height = 250, 350  # 设置输出图像的宽度和高度
    pts1 = np.float32([[left,x1], [high,y1], [low,y2], [right,x2]])
    pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])  # 定义透视变换后的角点位置

    matrix = cv2.getPerspectiveTransform(pts1, pts2)  # 获取透视变换矩阵
    imgOutput = cv2.warpPerspective(img, matrix, (width, height))  # 进行透视变换

    cv2.imshow("Output", imgOutput)

    cv2.imshow("Image", img)
    cv2.imshow("Output", imgOutput)
    cv2.imshow('Image', img)
    cv2.imshow("img",img)
    cv2.imshow("Stacked Images", imgStack)
    cv2.waitKey(1)
