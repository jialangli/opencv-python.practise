import cv2
import numpy as np

frameWidth = 640
frameHeight = 480
cap = cv2.VideoCapture(0)
cap=cv2.VideoCapture("LSP.mp4")
cap.set(3, frameWidth)
cap.set(4, frameHeight)
cap.set(10,150)

#长宽设置
widthimg,heightimg=540,840

def preProcessing(img):     #图像预处理，将处理完的放回
    Gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)       #转为灰度图来处理边缘
    Blur=cv2.GaussianBlur(Gray,(5,5),2)         #高斯模糊扩大误差
    canny= cv2.Canny(Blur,50,50)
    kernel=np.ones((5,5))               #以下是扩大边缘
    Dialation=cv2.dilate(canny,kernel,iterations=2)        #膨胀边缘
    imgThres=cv2.erode(Dialation,kernel,iterations=1)        #侵蚀,迭代次数
    return imgThres


def getcontours(img):  # 用于获取外形轮廓的函数
    biggest=np.array([])
    maxArea = 0
    countours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)  # 旧版三个参数
    # mode取值“CV_RETR_EXTERNAL”，method取值“CV_CHAIN_APPROX_NONE”，即只检测最外层轮廓，并且保存轮廓上所有点
    for cnt in countours:  # 取每一个外形轮廓
        area = cv2.contourArea(cnt)  # 面积
        if area>2000:  # 面积大于2000输出
            # cv2.drawContours(imgContour, cnt, -1, (0, 0, 255), 3)  # 画在imgContour上面
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)       #进行多边形拟合
            if area >maxArea and len(approx) ==4:           #迭代找到最大矩阵
                maxArea=area        #面积
                biggest = approx  # 角点
    cv2.drawContours(imgContour, biggest, -1, (0, 0, 255), 20)
    return biggest



def reorder(mypoints):      #鸟瞰函数(小点左上，大点右上)
    mypoints=mypoints.reshape((4,2))
    mypointsnew = np.zeros((4,1,2),np.int32)
    add = mypoints.sum(1)
    print("add",add)

    mypointsnew[0]=mypoints[np.argmin(add)]
    mypointsnew[3] = mypoints[np.argmax(add)]
    diff=np.diff(mypoints,axis=1)
    mypointsnew[1]=mypoints[np.argmin(diff)]
    mypointsnew[2] = mypoints[np.argmax(diff)]
    # print("newpoins",mypointsnew)
    return mypointsnew


def getWarp(img,biggest):       #获得鸟瞰图
    biggest=reorder(biggest)
    pts1 = np.float32(biggest)
    pts2 = np.float32([[0, 0], [widthimg, 0], [0, heightimg], [widthimg, heightimg]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)        #透视函数
    imgOutput = cv2.warpPerspective(img, matrix, (widthimg, heightimg))
    #裁剪画面
    imgCropped = imgOutput[20:imgOutput.shape[0]-20,20:imgOutput.shape[1]-20]
    imgCropped = cv2.resize(imgCropped,(widthimg,heightimg))

    return imgCropped

def stackImages(scale,imgArray):        #图像拼接
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

while True:
    success,img=cap.read()

    # img=cv2.imread("1211.jpg")
    img = cv2.resize(img,(widthimg,heightimg))
    imgContour=img.copy()

    imgThres=preProcessing(img)
    biggest=getcontours(imgThres)

    if biggest.size !=0:
        imgWarped=getWarp(img,biggest)
        imgWarped = cv2.filter2D(imgWarped, -2, kernel=1.1)
        imageArray = ([img,imgThres],
                  [imgContour,imgWarped])
    else:
        imageArray = ([img, imgThres],
                      [img, img])
    stackedImages = stackImages(0.6, imageArray)
    cv2.imshow("WorkFlow", stackedImages)
    cv2.waitKey(120)


    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break