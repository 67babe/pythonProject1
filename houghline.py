import cv2
import numpy as np
import zdyf
import math

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

#二值化
img=cv2.imread('img/std359.jpg')

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

closing=cv2.Canny(closing,45,50)
cv_show('img',closing)

linelist=[]

lines = cv2.HoughLinesP(closing, 0.8, np.pi / 720, 20,minLineLength=50, maxLineGap=20)
for line in lines:
    x1, y1, x2, y2 = line[0]
    linelist.append([x1,y1,x2,y2])
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)

linelist.sort(key=takesecond)

x1,y1,x2,y2=linelist[0]

cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1, lineType=cv2.LINE_AA)
cv_show("img",img)
tan=(y1-y2)/(x2-x1)
t=math.degrees(math.atan(tan))
print(t)

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


