import cv2
import numpy as np
import zdyf
import math

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

stdv=185.6765734265734

#二值化
img=cv2.imread('img/std268.jpg')

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('img',img_gray)
#zdyf.canny_trackbar(img_gray)
# cannyres=cv2.Canny(img_gray,40,120)

ret, h1 = cv2.threshold(img_gray.copy(), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
print("二值化")
cv_show('img',h1)
h1=cv2.bitwise_not(h1)
cv_show('img',h1)

#开运算
kernel = np.ones((3,3),np.uint8)
closing = cv2.morphologyEx(h1, cv2.MORPH_OPEN, kernel)
print("开运算")
cv2.imshow('opening', closing)
cv2.waitKey(0)
cv2.destroyAllWindows()

border=cv2.Canny(closing,45,50)
cv_show('img',closing)

linelist=[]

lines = cv2.HoughLinesP(border, 0.8, np.pi / 720, 20,minLineLength=50, maxLineGap=20)
for line in lines:
    x1, y1, x2, y2 = line[0]
    linelist.append([x1,y1,x2,y2])


linelist.sort(key=zdyf.takesecond)
print(linelist)
x1,y1,x2,y2=linelist[0]


cv_show("img",img)
tan=(y1-y2)/(x2-x1)
t=math.degrees(math.atan(tan))
print(t)

y=img.shape[0]
x=img.shape[1]
xmid=(int)(x/2)
ymid=(int)((linelist[0][2]+linelist[len(linelist)-1][2])/2)

M = cv2.getRotationMatrix2D((xmid, ymid), -t, 1)
closing = cv2.warpAffine(closing, M, (x, y))
img=cv2.warpAffine(img, M, (x, y))
cv2.imshow('rotation', closing)
cv2.waitKey(0)

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
draw_img = img.copy()
res=cv2.drawContours(draw_img,contours,-1,(0,0,255),1)
print("轮廓")
cv_show('img',res)



#T线轮廓矩形
cnt=contours[1]
tx,ty,tw,th = cv2.boundingRect(cnt)
tx1=tx+(int)(tw/4)
ty1=ty+(int)(th/10)
tx2=tx+tw-(int)(tw/4)
ty2=ty-(int)(th/3.5)+th
imgc = cv2.rectangle(img.copy(),(tx1,ty1),(tx2,ty2),(0,255,0),1)
print("T线轮廓矩形")
cv_show('img',img)

#T线分割
roi2=img.copy()[ty1:ty2,tx1:tx2]
print("T线分割")
cv_show('img',roi2)

#T线图像转灰度图
roi2_gray=cv2.cvtColor(roi2.copy(),cv2.COLOR_BGR2GRAY)
print("T线灰度图")
cv_show('img',roi2_gray)


#C线轮廓矩形
cnt=contours[0]
cx,cy,cw,ch = cv2.boundingRect(cnt)
cx1=cx+(int)(tw/4)
cy1=cy+(int)(th/10)
cx2=cx+cw-(int)(tw/4)
cy2=cy-(int)(th/3.5)+th
imgc = cv2.rectangle(imgc.copy(),(cx1,cy1),(cx2,cy2),(0,255,0),1)
print("C线轮廓矩形")
cv_show('img',imgc)

#C线分割
roi1=img.copy()[cy1:cy2,cx1:cx2]
print("C线分割")
cv_show('img',roi1)

#C线转灰度
roi1_gray=cv2.cvtColor(roi1.copy(),cv2.COLOR_BGR2GRAY)
print("C线图像转化为灰度图")
cv_show('img',roi1_gray)



#计算标C灰度图平均灰度值和方差
avgC=zdyf.cvAvg(roi1_gray)
variance=zdyf.Variance(roi1_gray)
print("平均灰度值为",avgC)
print("方差为",variance)


#w区域
wx=tx
wy=(int)((cy+ty)/2)
wh=th
ww=tw

wx1=cx1
wy1=(int)((cy1+ty1)/2)
wx2=cx2
wy2=wy1+cy2-cy1

imgc = cv2.rectangle(imgc.copy(),(wx1,wy1),(wx2,wy2),(0,255,0),1)
print("w区域轮廓矩形")
cv_show('img',imgc)
roiw=img.copy()[wy1:wy2,wx1:wx2]
print("w区域分割")
cv_show('img',roiw)
roiw_gray=cv2.cvtColor(roiw,cv2.COLOR_BGR2GRAY)
avgW=zdyf.cvAvg(roiw_gray)
print("W灰度为",avgW)
roiwhsv=cv2.cvtColor(roiw,cv2.COLOR_BGR2HSV)
roiwh,roiws,roiwv=cv2.split(roiwhsv)
avgWh=zdyf.cvAvg(roiwv)
print("w平均V分量",avgWh)


avgT=zdyf.cvAvg(roi2_gray)
variance2=zdyf.Variance(roi2_gray)
print("平均灰度值为",avgT)
print("方差为",variance2)

rate=avgC/avgT
print("灰度的比值为",rate)
result=(avgW-avgT)/((avgW-avgT)+(avgW-avgC))
print("T/(T+C)=",result)

# a=-0.0014
# b=0.0603
# c=0.1323-result
# p=np.poly1d([a,b,c])
# x1,x2=zdyf.solve_quad(a,b,c)
# print("x1=",x1,"x2=",x2)
# print(p.r)

