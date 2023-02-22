# -*- coding:utf-8 -*-
import cv2
from sys import argv
import serial
import threading
import time


def listener():
    while 1:
        str=ser.readline().decode("gbk")
        print(str)
        if(str=="start"):
            st=1
        if(str=="end"):
            st=0
        if(str=="pos"):
            st=2


def boardcast(x,y):
    stri=x[0:5]+"*"+y[0:5]+"!"
    ser.write(stri.encode("gbk"))


def CatchUsbVideo(window_name, camera_idx):
    bord = 25
    cah = 240
    caw = 320
    wt = 100
    while 1:
        if(st==1):break
    cv2.namedWindow(window_name)
    # 视频来源，可以来自一段已存好的视频，也可以直接来自USB摄像头
    cap = cv2.VideoCapture(camera_idx)
    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("./haarcascades/haarcascade_frontalface_alt2.xml")
    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    t = threading.Thread(target=listener)
    t.start()
    while cap.isOpened():
        if st==0 :break
        ok, frame = cap.read()  # 读取一帧数据
        if not ok:
            break
            # 将当前帧转换成灰度图像
        frame = cv2.resize(frame, (caw, cah))
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
        faceRects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
        rows, cols = grey.shape
        wm = -1
        hm = -1
        xm = -1
        ym = -1
        if len(faceRects) > 0:  # 大于0则检测到人脸
            for faceRect in faceRects:  # 单独框出每一张人脸
                x, y, w, h = faceRect
                if (w > wm and h > hm):
                    wm = w
                    hm = h
                    xm = x
                    ym = y
        x = xm
        y = ym
        w = wm
        h = hm
        if (x != -1 and y != -1): cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
        if (x != -1 and y != -1 and (x < bord or x + w > cols - bord or y < bord or y + h > rows - bord)):
            # print("you need change"+str(x+w)+" "+str(y+h)+" "+str(rows)+" "+str(cols))
            # 显示图像
            t2 = threading.Thread(target=boardcast,args=(str(int(x + w / 2) / cols) ,str(int(y + h / 2) / rows)) )
            t2.start()
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(wt)
        else:
            # 显示图像
            # print(ser.readline().decode("gbk"))
            cv2.imshow(window_name, frame)
            c = cv2.waitKey(1)
            # if c & 0xFF == ord('q'):
            # 判断是否点击了右上角的关闭按钮
        if cv2.getWindowProperty(window_name, 0) == -1:
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    ser.close()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    ser = serial.Serial("/dev/ttyAMA0", 9600, timeout=0.1)
    st=0
    if (ser.isOpen == False):
        ser.open()
    if len(argv) != 1:
        print("Usage:%s camera_id\r\n" % (argv[0]))
    else:
        CatchUsbVideo('Face Detection Example Program', 0)

