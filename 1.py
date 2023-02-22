import cv2
import numpy as np
from matplotlib import pyplot as plt

# 17.分水岭算法








'''
# 16.反向投影 & 模板匹配
cap=cv2.VideoCapture(0)
ret,img=cap.read()
imgtar=cv2.imread("1.jpg")
img2=img;img=imgtar;imgtar=img2
imgtarhsv=cv2.cvtColor(imgtar,cv2.COLOR_BGR2HSV)
imghsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
rh=cv2.calcHist([imghsv],[0,1],None,[180,256],[0,180,0,256])
cv2.normalize(rh,rh,0,255,cv2.NORM_MINMAX)
dst=cv2.calcBackProject([imgtarhsv],[0,1],rh,[0,180,0,256],1)
thresh=cv2.merge((dst,dst,dst))
cv2.imshow("bef",imgtar)
res=cv2.bitwise_and(imgtar,thresh)
cv2.imshow("aft",res)
img2=cv2.matchTemplate(imgtar,img,cv2.TM_CCOEFF_NORMED)
w,h,c=img.shape[::]
miv,mav,mil,mal=cv2.minMaxLoc(img2)
tl=mal
br=(tl[0]+w,tl[1]+h)

cv2.rectangle(imgtar,tl,br,(255,255,255),5)
cv2.imshow("aft2",imgtar)
cv2.waitKey(0)
'''

"""
# 15.图像的矩
img=cv2.imread("4.jpg",0)
rows,cols=img.shape
a=[]
for i in range(0,rows):
    for j in range(0,cols):
        if img[i,j]>200:
            a.append([j,i])
a=np.array(a)
M=cv2.moments(a)
print(M)
"""

'''
# 14.直线拟合 & 多边形拟合
img=cv2.imread("4.jpg",0)
cv2.imshow("bef",img)
img2=img
rows,cols=img2.shape
a=[]
for i in range(0,rows):
    for j in range(0,cols):
        if img[i,j]>200:
            a.append([j,i])
a=np.array(a)
[vx,vy,x,y]=cv2.fitLine(a,cv2.DIST_L2,0,0.01,0.01)
lef=int((-x*vy/vx)+y)
rig=int(((cols-x)*vy/vx)+y)
img2=cv2.line(img2,(cols-1,rig),(0,lef),100,3)
cv2.imshow("aft1",img2)
poly=cv2.approxPolyDP(a,1,False)
print(poly)
xa=()
for l in poly:
    l=(l[0][0],l[0][1])
    if xa!=():
        cv2.line(img,xa,l,255,4)
    xa=l
cv2.imshow("aft2",img)
cv2.waitKey(0)
'''

"""
# 13.轮廓与凸包
cap = cv2.VideoCapture(0)
while 1:
    ok, imgc = cap.read()
    img=cv2.cvtColor(imgc,cv2.COLOR_BGR2GRAY)
    # 轮廓拟合
    img2=img
    img2=cv2.Canny(img2,190,250)
    cv2.imshow("before", img2)
    con,hi=cv2.findContours(img2,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(img2,con,-1,255,2)
    # cv2.imshow("aft1",img2)
    # 凸包查找 & 边界矩形
    for c in con:
        # if cv2.isContourConvex(c):cv2.drawContours(img3,[c],-1,255,3)
        rect=cv2.minAreaRect(c)
        box=cv2.boxPoints(rect)
        img4=cv2.line(imgc,tuple(box[0]),tuple(box[1]),(255,0,0,0),2)
        img4 = cv2.line(imgc, tuple(box[1]), tuple(box[2]), (255, 0, 0, 0), 2)
        img4 = cv2.line(imgc, tuple(box[2]), tuple(box[3]), (255, 0, 0, 0), 2)
        img4 = cv2.line(imgc, tuple(box[0]), tuple(box[3]), (255, 0, 0, 0), 2)
    # cv2.imshow("aft2", img3)
    cv2.imshow("aft3", img4)

    cv2.waitKey(1)
"""

'''
# 12.一维直方图
cap = cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    cv2.imshow("before", img)
    # 直方图展示
    hist=cv2.calcHist([img],[0],None,[256],[0,256])
    plt.plot(hist,color='b')
    plt.xlim([0,256])
    # plt.show()
    # 直方图均衡化
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    equ=cv2.equalizeHist(img)
    # cv2.imshow("after1",equ)
    # img=np.hstack((img,equ))
    # cv2.imshow("after2", img)
    # 自适应直方图均衡化
    clahe=cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    img=clahe.apply(img)
    cv2.imshow("after3", img)
    cv2.waitKey(1)
'''

"""
# 11.重映射
cap = cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    cv2.imshow("before", img)
    map1=np.zeros((2000,2000),np.float32)
    map2 = np.zeros((2000, 2000), np.float32)
    for i in range(0,2000):
        map2[i,:]=i
        map1[:,i]=i
    img=cv2.remap(img,map1,map2,cv2.INTER_LINEAR)
    cv2.imshow("aft", img)
    cv2.waitKey(1)
"""

'''
# 10.霍夫变换
cap = cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    cv2.imshow("before", img)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img=cv2.Canny(img,120,170,apertureSize=3)
    cv2.imshow("canny", img)
    """
    # 霍夫线变换
    lines=cv2.HoughLines(img,1,np.pi/180,160)
    for line in lines:
        for rho,theta in line:
            a=np.cos(theta);b=np.sin(theta)
            x0=a*rho;y0=b*rho
            x1=int(x0+1000*(-b))
            x2=int(x0-1000*(-b))
            y1=int(y0+1000*a)
            y2=int(y0-1000*a)
            cv2.line(img,(x1,y1),(x2,y2),255,4)
    cv2.imshow("huoghlines", img)
    cv2.waitKey(1)
    """
    """
    # 霍夫圆变换
    img=cv2.bilateralFilter(img,3,300,50)
    circles=cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,dp=3,minDist=100,param1=150,param2=450)
    if circles is not None:
        for cir in circles:
            for i in cir:
                cv2.circle(img,(i[0],i[1]),i[2],255,3)
    cv2.imshow("houghcircles",img)
    cv2.waitKey(1)
    """
'''

"""
# 9.图像梯度
cap = cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    cv2.imshow("before", img)
    lap=cv2.Laplacian(img,cv2.CV_64F)
    # lap=cv2.normalize(lap,0,255,cv2.NORM_MINMAX)
    lap=np.absolute(lap)
    lap=np.uint8(lap)
    lap2=cv2.Laplacian(img,-1)
    sobelx=cv2.Sobel(img,-1,1,0,5)
    sobely = cv2.Sobel(img, -1, 0, 1, 5)
    cv2.imshow("Laplacian", lap)
    cv2.imshow("Laplacian2", lap2)
    # cv2.imshow("sobelx", sobelx)
    # cv2.imshow("sobely", sobely)
    cv2.waitKey(1)
"""

'''
# 8.漫水填充
cap = cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    cv2.imshow("before", img)
    h, w = img.shape[:2]
    mask = np.zeros([h + 2, w + 2], np.uint8)
    cv2.floodFill(img,mask,(50,300),(155,255,55),(10,10,10),(10,10,10),flags=cv2.FLOODFILL_FIXED_RANGE)
    cv2.imshow("aft",img)
    cv2.waitKey(1)
'''

"""
# 7.形态学方法
cap = cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    cv2.imshow("before", img)
    ker = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img2 = cv2.dilate(img, ker)  # 膨胀
    # cv2.imshow("dilate",img2)
    img3 = cv2.erode(img, None)  # 腐蚀
    # cv2.imshow("erode", img3)
    # img2=cv2.erode(img2,None)
    # img2=cv2.morphologyEx(img,cv2.MORPH_OPEN,ker)
    # img3=cv2.dilate(img3,None)
    # img2 = cv2.morphologyEx(img, cv2.MORPH_CLOSE, ker)
    # cv2.imshow("close", img2)
    # cv2.imshow("open", img3)
    # img4=img2-img3
    # cv2.imshow("edge", img4)
    img4 = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, ker)  # 礼帽
    cv2.imshow("tophat", img4)
    img4 = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, ker)  # 黑帽
    cv2.imshow("blackhat", img4)
    cv2.waitKey(1)
"""

'''
# 6.滤波
cap = cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    cv2.imshow("before", img)
    img2 = cv2.boxFilter(img, -1, (5, 5), (-1, -1))  # 方框滤波
    # cv2.imshow("common blur", img2)
    img2 = cv2.GaussianBlur(img, (5, 5),0,0)  # 高斯滤波
    # cv2.imshow("gussian blur", img2)
    img2 = cv2.medianBlur(img, 5)  # 中值滤波
    # cv2.imshow("median blur", img2)
    img2 = cv2.bilateralFilter(img, 5,30,10)  # 双边滤波
    # cv2.imshow("bilateral blur", img2)
    cv2.waitKey(1)
'''

"""
# XML、YAML文件与FileStorage
fw=cv2.FileStorage("1.yml",cv2.FileStorage_WRITE)
fw.write("haha",12345)
fw.release()
fr=cv2.FileStorage("1.yml",cv2.FileStorage_READ)
num=fr.getNode("haha").real()
print(num)
fr.release()
"""

'''
# 5.傅里叶变换
cap = cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    cv2.imshow("before", img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 灰度转换
    img2 = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)  # 傅里叶变换
    img2 = np.fft.fftshift(img2)  # 图像中心移动
    img3 = 20 * np.log(cv2.magnitude(img2[:, :, 0], img2[:, :, 1]))
    # plt.imshow(img4,cmap='gray')
    # plt.show()
    img3 = cv2.normalize(img3, 0, 255, cv2.NORM_MINMAX)  # 归一化
    cv2.imshow("aft2", img3)
    rows, cols = img.shape
    crow, ccol = int(rows / 2), int(cols / 2)
    # mask = np.zeros((rows, cols, 2), np.uint8)  # 创建掩码,低通滤波
    # mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1
    # img4 = img2 * mask
    img4=img2
    img4[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0 # 高通滤波
    img4 = np.fft.ifftshift(img4)
    img5 = cv2.idft(img4)
    img5 = cv2.magnitude(img5[:, :, 0], img5[:, :, 1])
    img5 = cv2.normalize(img5, 0, 255, cv2.NORM_MINMAX)  # 归一化
    cv2.imshow("high", img5)
    cv2.waitKey(1)
'''

"""
# 4.记录两次按下q键的时间间隔 && 阈值化 && 图像变换
cap=cv2.VideoCapture(0)
t0=-1
while 1:
    ok,img=cap.read()
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)                #单通道化
    # img=cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)   # 自适应阈值化
    # ret,img=cv2.threshold(img,127,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)  # 简单阈值化
    # img=cv2.resize(img,None,2,2,cv2.INTER_CUBIC)          # 图像的变大缩小
    # move=np.array([[1,0,100],[0,1,50]],np.float32)        # 图像的平移矩阵
    # img=cv2.warpAffine(img,move,(1024,768))               # 图像的平移
    # rows,cols=img.shape                                   # 获取高和宽
    # M=cv2.getRotationMatrix2D((cols/2,rows/2),45,1)       # 图像的旋转矩阵
    # img=cv2.warpAffine(img,M,(cols,rows))                 # 图像的旋转
    # pst1=np.float32([[200,0],[0,0],[0,200],[200,200]])
    # pst2=np.float32([[250,50],[0,0],[0,250],[250,300]])
    # M=cv2.getPerspectiveTransform(pst1,pst2)              # 透视变换
    # img = cv2.warpPerspective(img, M, (2000, 1000))
    # M=cv2.getAffineTransform(pst1,pst2)                   # 仿射变换
    # img=cv2.warpAffine(img,M,(300,300))
    cv2.imshow("1",img)
    if cv2.waitKey(1)& 0xFF == ord('q'):
        t1=cv2.getTickCount()
        if t0 !=-1:
            print((t1-t0)/cv2.getTickFrequency())
        t0=t1
"""

'''
# 3.单击显示名字
def onMouse(event,x,y,flags,para):
    if(event==1):
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,"LJQ",(x,y),font,1,255,2)
        cv2.imshow("1", img)
        cv2.waitKey(0)


img=cv2.imread("1.jpg")
cv2.imshow("1",img)
cv2.setMouseCallback("1",onMouse,5)
cv2.waitKey(0)
'''

"""
# 2.多张图片根据滑动条进行融合
def onChange(pos):
    print(pos)
    g_alphaVala=pos/100
    g_alphaValb=1-g_alphaVala
    img3=cv2.addWeighted(img1,g_alphaVala,img2,g_alphaValb,0.0)
    cv2.imshow("1",img3)


img1=cv2.imread("1.jpg")
img2=np.zeros((178,178,3),np.uint8)
g_default=70
cv2.namedWindow("1",2)
cv2.createTrackbar("2","1",g_default,100,onChange)
onChange(g_default)
cv2.waitKey(0)
"""

"""
# 1.视频的Canny边缘检测
cap = cv2.VideoCapture("2.mp4")
# cap=cv2.VideoCapture(0)
while 1:
    ok, img = cap.read()
    if ok == 0: break
    cv2.imshow("before", img)
    img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img3 = cv2.blur(img2, (5, 5))
    img3 = cv2.Canny(img3, 40, 60)
    cv2.imshow("after", img3)
    cv2.waitKey(25)
    # cv2.imwrite("2.jpg",img3)


cap.release()
"""
