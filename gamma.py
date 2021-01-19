import cv2

def cv_show(name,img):
    cv2.imshow(name,img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def gammaadj(img,r):
    for x in range(len(img)):
        for x2 in range(len(img[x])):
            img[x][x2]=(int)((img[x][x2]/255)**r*255)
    return img

img=cv2.imread("img/1.jpg")
cv_show("img",img)
img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv_show("img",img)
gammaadj(img,0.5)
cv_show('img',img)