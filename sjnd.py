import cv2
import numpy as np


def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def cvAvg(img):
    sum=0

    for y in img:
        for x in y:
            sum=sum+x
    return sum/(len(img)*len(img[0]))

img=cv2.imread('img/sj.jpg')
img2=cv2.imread('img/shizhi.jpg')
# hsv=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV_FULL)
# h,s,v=cv2.split(hsv)
# ret, h1 = cv2.threshold(h, 235, 255, cv2.THRESH_BINARY)
# kernel = np.ones((3,3),np.uint8)
# erosion = cv2.erode(h1,kernel,iterations = 4)
# blur = cv2.boxFilter(img2,-1,(3,3), normalize=True)
# 
# cv2.imshow('blur', blur)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#边界填充
top_size,bottom_size,left_size,right_size = (40,40,40,40)
img2 = cv2.copyMakeBorder(img2, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
print("边界填充")
cv_show('img',img2)

#转化为HSV
hsv=cv2.cvtColor(img2,cv2.COLOR_BGR2HSV_FULL)
h,s,v=cv2.split(hsv)
print("转化为HSV,H分量图")
cv_show('img',h)

#二值化
ret, h1 = cv2.threshold(h.copy(), 240, 255, cv2.THRESH_BINARY)
print("二值化")
cv_show('img',h1)

#开运算
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(h1, cv2.MORPH_OPEN, kernel)
print("开运算")
cv2.imshow('opening', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

#闭运算
kernel = np.ones((10,10),np.uint8)
closing = cv2.morphologyEx(closing, cv2.MORPH_CLOSE, kernel)
print("闭运算")
cv2.imshow('closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

#开运算
kernel = np.ones((10,10),np.uint8)
closing = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
print("开运算")
cv2.imshow('opening', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()



#开运算
kernel = np.ones((50,50),np.uint8)
closing = cv2.morphologyEx(closing.copy(), cv2.MORPH_CLOSE, kernel)
print("闭运算")
cv2.imshow('closing', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()


# #腐蚀
# kernel = np.ones((4,4),np.uint8)
# erosion = cv2.erode(closing,kernel,iterations = 2)
# print("腐蚀")
# cv_show('mig',erosion)

#轮廓
contours, hierarchy = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
draw_img = img2.copy()
res=cv2.drawContours(draw_img,contours,-1,(0,0,255),1)
print("轮廓")
cv_show('img',res)
   
#标准线轮廓矩形
cnt=contours[0]
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img2.copy(),(x+17,y+4),(x-17+w,y-7+h),(0,255,0),1)
print("标准线轮廓矩形")
cv_show('img',img)

#标准线分割
roi1=img2.copy()[y+4:y-7+h,x+17:x-17+w]
print("标准线分割")
cv_show('img',roi1)

#标准线转灰度
roi1_gray=cv2.cvtColor(roi1.copy(),cv2.COLOR_BGR2GRAY)
print("标准线图像转化为灰度图")
cv_show('img',roi1_gray)

#计算标准线灰度图平均灰度值
avg1=cvAvg(roi1_gray)
print("平均灰度值为",avg1)

#检测线轮廓矩形
cnt=contours[1]
x,y,w,h = cv2.boundingRect(cnt)
img = cv2.rectangle(img2.copy(),(x+17,y+4),(x-17+w,y-7+h),(0,255,0),1)
print("检测线轮廓矩形")
cv_show('img',img)

#检测线分割
roi2=img2.copy()[y+4:y-7+h,x+17:x-17+w]
print("检测线分割")
cv_show('img',roi2)

#检测线图像转灰度图
roi2_gray=cv2.cvtColor(roi2.copy(),cv2.COLOR_BGR2GRAY)
print("检测线灰度图")
cv_show('img',roi2_gray)

avg2=cvAvg(roi2_gray)
print("平均灰度值为",avg2)


rate=avg1/avg2
print("灰度的比值为",rate)
