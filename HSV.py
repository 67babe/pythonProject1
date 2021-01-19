import cv2
import numpy as np

img=cv2.imread('img/IMG_2564.JPG')
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Hsv_adj(img,num):
    hsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v=cv2.split(hsv)
    for y in range(len(v)):
        for x in range(len(v[y])):
            if(v[y][x]+num>255):
                v[y][x]=205
            else:
                v[y][x] = v[y][x]+num
    img=cv2.merge((h,s,v))
    img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
    return img


img=cv2.resize(img,(0,0),fx=0.2,fy=0.2)
res = Hsv_adj(img,40)
cv_show("img",res)