import cv2
import numpy as np

def canny_trackbar(img):
    cv2.namedWindow('Canny')

    # 定义回调函数
    def nothing(x):
        pass

    # 创建两个滑动条，分别控制threshold1，threshold2
    cv2.createTrackbar('threshold1', 'Canny', 0, 400, nothing)
    cv2.createTrackbar('threshold2', 'Canny', 0, 400, nothing)
    while (1):
        # 返回滑动条所在位置的值
        threshold1 = cv2.getTrackbarPos('threshold1', 'Canny')
        threshold2 = cv2.getTrackbarPos('threshold2', 'Canny')
        # Canny边缘检测
        img_edges = cv2.Canny(img, threshold1, threshold2)
        # 显示图片
        cv2.imshow('original', img)
        cv2.imshow('Canny', img_edges)
        if cv2.waitKey(1) == ord('q'):
            break
    cv2.destroyAllWindows()

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

def threshold_trackbar(img):
    cv2.namedWindow('threshold')

    # 定义回调函数
    def nothing(x):
        pass

    # 创建两个滑动条，分别控制threshold1，threshold2
    cv2.createTrackbar('threshold1', 'threshold', 0, 400, nothing)
    cv2.createTrackbar('threshold2', 'threshold', 0, 400, nothing)
    while (1):
        # 返回滑动条所在位置的值
        threshold1 = cv2.getTrackbarPos('threshold1', 'threshold')
        threshold2 = cv2.getTrackbarPos('threshold2', 'threshold')
        # Canny边缘检测
        ret,h1 = cv2.threshold(img, threshold1, threshold2,cv2.THRESH_BINARY)
        # 显示图片
        cv2.imshow('original', img)
        cv2.imshow('threshold', h1)
        if cv2.waitKey(1) == ord('q'):
            break
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

def cvAvg(img):
    sum=0

    for y in img:
        for x in y:
            sum=sum+x
    return sum/(len(img)*len(img[0]))

def cvContrastRatio(img,num):
    for y in img:
        for x in y:
            x[0] = x[0] * num
            x[1] = x[1] * num
            x[2] = x[2] * num
            if (x[0] > 255):
                x[0] = 255
            if (x[1] > 255):
                x[1] = 255
            if (x[2] > 255):
                x[2] = 255
    return img

def Variance(img):
    sum=0
    avg=cvAvg(img)
    for y in img:
        for x in y:
            sum = sum + (x-avg)*(x-avg)
    return (int)(sum/(len(img)*len(img[0])))

def takesecond(elem):
    return elem[1]

def gammaAdj(img,r):
    for x in range(len(img)):
        for x2 in range(len(img[x])):
            img[x][x2]=(int)((((img[x][x2]/255)**(0.45))*r)**(2.2)*255)
    return img