import cv2
import numpy as np
import Exposure

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

#试试git 再试试——仔试试
def solve_quad(a,b,c):
    if a == 0:
        print('您输入的不是二次方程!')
    else:
        delta = b*b-4*a*c
        x = -b/(2*a)
        if delta == 0:
            print('方程有惟一解，X=%f'%(x))
            return x
        elif delta > 0:
            x1 = x-np.sqrt(delta)/(2*a)
            x2 = x+np.sqrt(delta)/(2*a)
            print('方程有两个实根:X1=%f,X2=%f'%(x1,x2))
            return x1,x2
        else:
            x1 = (-b+complex(0,1)*np.sqrt((-1)*delta))/(2*a)
            x2 = (-b-complex(0,1)*np.sqrt((-1)*delta))/(2*a)
            print('方程有两个虚根，如下所示：')
            print(x1,x2)
            return x1,x2

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

def cvContrastRatio(img,num):
    b,g,r=cv2.split(img)

    for y in range(len(r)):
        for x in range(len(r[y])):
            r[y][x]=(int)(r[y][x]*num)
    for y in range(len(g)):
        for x in range(len(g[y])):
            g[y][x] = (int)(g[y][x] * num)
    for y in range(len(b)):
        for x in range(len(b[y])):
            b[y][x] = (int)(b[y][x]* num )
    res=cv2.merge((b,g,r))
    return res

def Variance(img):
    sum=0
    avg=cvAvg(img)
    for y in img:
        for x in y:
            sum = sum + (x-avg)*(x-avg)
    return (int)(sum/(len(img)*len(img[0])))

img=cv2.imread('img/2w.jpg')
img=cvContrastRatio(img.copy(),1.4)

##模式匹配
template=cv2.imread('img/templatetest.jpg')
res = cv2.matchTemplate(img.copy(), template, cv2.TM_CCOEFF_NORMED)
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
xmid=template.shape[1]
print(xmid)

ymid=template.shape[0]
print(ymid)
raw=img.copy()[max_loc[1]:max_loc[1]+template.shape[0],max_loc[0]:max_loc[0]+template.shape[1]]
mres=img.copy()[max_loc[1]+(int)(ymid/8):max_loc[1]+template.shape[0]-(int)(ymid/8),max_loc[0]+(int)(xmid/3):max_loc[0]+template.shape[1]-(int)(xmid/3)]
print("模式匹配")
cv_show("mres",mres)

#对比度
##mresc=cvContrastRatio(mres,4)
#cv_show("mres",mresc)

mres=cv2.bilateralFilter(mres,21,10,10)

cv_show("mres",mres)
mres_gray=cv2.cvtColor(mres,cv2.COLOR_BGR2GRAY)
cv_show("mres_gray",mres_gray)
bord=cv2.Canny(mres_gray,25,18)
cv_show("b",bord)

lines=cv2.HoughLinesP(mres_gray,1,np.pi/10,50,5,10)
for line in lines:
    for x1,y1,x2,y2 in line:
        if abs(x1-x2)>=template.shape[1]/5 and abs(y1-y2) <= 4:
            cv2.line(mres_gray,(x1,y1),(x2,y2),[255,0,0],1)

cv_show("lines",mres_gray)
#边界填充
top_size,bottom_size,left_size,right_size = ((int)(ymid/8),(int)(ymid/8),(int)(xmid/3),(int)(xmid/3))
mres = cv2.copyMakeBorder(mres, top_size, bottom_size, left_size, right_size,cv2.BORDER_CONSTANT, value=0)
print("边界填充")
cv_show('img',mres)

#转化为HSV
hsv=cv2.cvtColor(mres,cv2.COLOR_BGR2HSV_FULL)
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
draw_img = raw.copy()
res=cv2.drawContours(draw_img,contours,-1,(0,0,255),1)
print("轮廓")
cv_show('img',res)

#C线轮廓矩形
cnt=contours[0]
cx,cy,cw,ch = cv2.boundingRect(cnt)
img = cv2.rectangle(raw.copy(),(cx,cy+9*(int)(ch/5)),(cx+cw,cy+ch+6*(int)(ch/5)),(0,255,0),1)
print("C线轮廓矩形")
cv_show('img',img)

#C线分割
roi1=raw.copy()[cy+(int)(ch/5):cy-(int)(ch/2)+ch,cx:cx+cw]
print("C线分割")
cv_show('img',roi1)

#C线转灰度
roi1_gray=cv2.cvtColor(roi1.copy(),cv2.COLOR_BGR2GRAY)
print("C线图像转化为灰度图")
cv_show('img',roi1_gray)



#计算标C灰度图平均灰度值和方差
avgC=cvAvg(roi1_gray)
variance=Variance(roi1_gray)
print("平均灰度值为",avgC)
print("方差为",variance)

#T线轮廓矩形
cnt=contours[1]
tx,ty,tw,th = cv2.boundingRect(cnt)
img = cv2.rectangle(img,(tx,ty-7*(int)(th/5)),(tx+tw,ty-10*(int)(th/5)+th),(0,255,0),1)
print("T线轮廓矩形")
cv_show('img',img)

#T线分割
roi2=raw.copy()[ty+(int)(th/5):ty-(int)(th/2)+th,tx:tx+tw]
print("T线分割")
cv_show('img',roi2)

#T线图像转灰度图
roi2_gray=cv2.cvtColor(roi2.copy(),cv2.COLOR_BGR2GRAY)
print("T线灰度图")
cv_show('img',roi2_gray)

#w区域
wx=cx
wy=(int)((cy+ty)/2)
wh=ch
ww=cw

img = cv2.rectangle(img,(wx,wy+(int)(wh/5)),(wx+ww,wy-(int)(wh/2)+wh),(0,255,0),1)
print("w区域轮廓矩形")
cv_show('img',img)
roiw=raw.copy()[wy+(int)(wh/5):wy-(int)(wh/2)+wh,wx:wx+ww]
print("w区域分割")
cv_show('img',roiw)
roiw_gray=cv2.cvtColor(roiw,cv2.COLOR_BGR2GRAY)
avgW=cvAvg(roiw_gray)
print("W灰度为",avgW)


avgT=cvAvg(roi2_gray)
variance2=Variance(roi2_gray)
print("平均灰度值为",avgT)
print("方差为",variance2)

rate=avgC/avgT
print("灰度的比值为",rate)

result=(avgW-avgT)/((avgW-avgT)+(avgW-avgC))
print("T/(T+C)=",result)

a=-0.0014
b=0.0603
c=0.1323-result
p=np.poly1d([a,b,c])
x1,x2=solve_quad(a,b,c)
print("x1=",x1,"x2=",x2)
print(p.r)

