import cv2
import numpy as np



def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



#二值化
img=cv2.imread('img/ykt.jpg')

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show('img',img_gray)
ret, h1 = cv2.threshold(img_gray.copy(), 75, 255, cv2.THRESH_BINARY)
print("二值化")
cv_show('img',h1)



