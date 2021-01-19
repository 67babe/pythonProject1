import cv2
import numpy as np

img=cv2.imread('img/IMG_2564.JPG')
def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
def Exposure_adj(img,num):
    b,g,r=cv2.split(img)

    for y in range(len(r)):
        for x in range(len(r[y])):
            r[y][x]=(int)((r[y][x]/255)**num*255)
    for y in range(len(g)):
        for x in range(len(g[y])):
            g[y][x] = (int)((g[y][x] / 255) ** num * 255)
    for y in range(len(b)):
        for x in range(len(b[y])):
            b[y][x] = (int)((b[y][x] / 255) ** num * 255)
    res=cv2.merge((b,g,r))
    return res


# img=cv2.resize(img,(0,0),fx=0.2,fy=0.2)
# res = Exposure_adj(img,0.55)
# cv_show("img",res)