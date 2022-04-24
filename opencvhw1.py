import sys
from Opencvdl_HW1_ui import Ui_window
import cv2
from PyQt5.QtWidgets import QMainWindow, QApplication
import numpy as np
import glob
from matplotlib import pyplot as plt

class MainWindow(QMainWindow, Ui_window):
    def __init__(self, parent=None):
        super(MainWindow, self).__init__(parent)
        self.setupUi(self)
        self.onBindingUI()

    # Write your code below
    # UI components are defined in Opencvdl_HW1_ui.py, please take a look.
    # You can also open Opencvdl_HW1.ui by qt-designer to check ui components.

    def onBindingUI(self):
        self.btn1_1.clicked.connect(self.on_btn1_1_click)
        self.btn1_2.clicked.connect(self.on_btn1_2_click)
        self.btn1_3.clicked.connect(self.on_btn1_3_click)
        self.btn1_4.clicked.connect(self.on_btn1_4_click)
        self.btn2_1.clicked.connect(self.on_btn2_1_click)
        self.btn2_2.clicked.connect(self.on_btn2_2_click)
        self.btn2_3.clicked.connect(self.on_btn2_3_click)
        self.btn3_1.clicked.connect(self.on_btn3_1_click)
        self.btn3_2.clicked.connect(self.on_btn3_2_click)
        self.btn3_3.clicked.connect(self.on_btn3_3_click)
        self.btn3_4.clicked.connect(self.on_btn3_4_click)
        self.btn4_1.clicked.connect(self.on_btn4_1_click)
        self.btn4_2.clicked.connect(self.on_btn4_2_click)
        self.btn4_3.clicked.connect(self.on_btn4_3_click)
        self.btn4_4.clicked.connect(self.on_btn4_4_click)

    def on_btn1_1_click(self):
        img = cv2.imread('.\\Data\\Q1_Image\\Sun.jpg')
        size = img.shape
        #size = (長,寬,通道數)
        print('height',size[0])
        print('weight',size[1])
        cv2.imshow('Sun',img)

    def on_btn1_2_click(self):
        img = cv2.imread('.\\Data\\Q1_Image\\Sun.jpg')
        b,g,r = cv2.split(img)
        #經過cv2.split之後，每個通道是768*1024的單通道圖像
        zeros = np.zeros(img.shape[:2], dtype = "uint8")

        cv2.namedWindow('Blue', cv2.WINDOW_NORMAL)
        cv2.imshow("Blue", cv2.merge([b, zeros, zeros]))
        cv2.namedWindow('Green', cv2.WINDOW_NORMAL)
        cv2.imshow("Green", cv2.merge([zeros, g, zeros]))
        cv2.namedWindow('Red', cv2.WINDOW_NORMAL)
        cv2.imshow("Red", cv2.merge([zeros, zeros, r]))

    def on_btn1_3_click(self):
        img = cv2.imread('.\\Data\\Q1_Image\\Sun.jpg')
        img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        b,g,r = cv2.split(img)
        mix = b//3 + g//3 + r//3
        image = cv2.merge([mix,mix,mix])   

        cv2.imshow('opencv',img1)
        cv2.imshow('average',image)


    def on_btn1_4_click(self):
        img1 = cv2.imread('.\\Data\\Q1_Image\\Dog_Strong.jpg')
        img2 = cv2.imread('.\\Data\\Q1_Image\\Dog_Weak.jpg')
        def on_change(x):
            w = cv2.getTrackbarPos('Blending', 'Blend')
            left = w/255
            right = 1 - left
            dst = cv2.addWeighted(img1, left , img2 , right , 0)
            cv2.imshow('Blend', dst)

        cv2.namedWindow('Blend')
        cv2.createTrackbar('Blending', 'Blend', 0, 255, on_change)

    def on_btn2_1_click(self):
        img = cv2.imread('.\\Data\\Q2_Image\\Lenna_whiteNoise.jpg')
        Gaussian = cv2.GaussianBlur(img,(5,5),0)
        cv2.namedWindow('Gaussian', cv2.WINDOW_NORMAL)
        cv2.imshow('Gaussian', Gaussian)

    def on_btn2_2_click(self):
        img = cv2.imread('.\\Data\\Q2_Image\\Lenna_whiteNoise.jpg')
        bilateral =  cv2.bilateralFilter(img,9,90,90)
        cv2.namedWindow('init', cv2.WINDOW_NORMAL)
        cv2.imshow('init', img)
        cv2.namedWindow('bilateral', cv2.WINDOW_NORMAL)
        cv2.imshow('bilateral', bilateral)

    def on_btn2_3_click(self):
        img = cv2.imread('.\\Data\\Q2_Image\\Lenna_pepperSalt.jpg')
        Median1 = cv2.medianBlur(src=img,ksize = 3 )
        cv2.namedWindow('init', cv2.WINDOW_NORMAL)
        cv2.imshow('init', img)
        cv2.namedWindow('Median 3x3', cv2.WINDOW_NORMAL)
        cv2.imshow('Median 3x3', Median1)
        Median2 = cv2.medianBlur(src=img,ksize = 5 )
        cv2.namedWindow('init', cv2.WINDOW_NORMAL)
        cv2.imshow('init', img)
        cv2.namedWindow('Median 5x5', cv2.WINDOW_NORMAL)
        cv2.imshow('Median 5x5', Median2)

    def on_btn3_1_click(self):
        img = cv2.imread('.\\Data\\Q3_Image\\House.jpg',cv2.IMREAD_GRAYSCALE)
        gaf= [[0.045, 0.122, 0.045], [0.122, 0.332,0.122], [0.045, 0.122, 0.045]]
        test=[[0, 0, 0], [0, 0,0], [0, 0, 0]]
        solx= [[-1, 0, 1], [-2, 0,2], [-1, 0, -1]]

        img2 = cv2.imread('.\\Data\\Q3_Image\\House.jpg',cv2.IMREAD_GRAYSCALE)
        #()
        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                img2[i][j]=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
        cv2.imshow('Gaussian Blur',img2)

    def on_btn3_2_click(self):
        img = cv2.imread('.\\Data\\Q3_Image\\House.jpg',cv2.IMREAD_GRAYSCALE)
        gaf= [[0.045, 0.122, 0.045], [0.122, 0.332,0.122], [0.045, 0.122, 0.045]]
        solx= [[-1, 0, 1], [-2, 0,2], [-1, 0, 1]]

        img2 = img.copy()

        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                img2[i][j]=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
        img = img2.copy()
        gaf = solx

        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                temp=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
                if temp > 255 :
                    temp = 255
                if temp < 0 :
                    temp = abs(temp)
                img2[i][j] = temp


        cv2.imshow('Sobel X',img2)

    def on_btn3_3_click(self):
        img = cv2.imread('.\\Data\\Q3_Image\\House.jpg',cv2.IMREAD_GRAYSCALE)
        gaf= [[0.045, 0.122, 0.045], [0.122, 0.332,0.122], [0.045, 0.122, 0.045]]
        solx= [[1,2,1], [0, 0,0], [-1, -2, -1]]

        temp = img.copy()
        #()
        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                temp[i][j]=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
        img = temp.copy()
        gaf = solx

        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                temp2=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
                if temp2 > 255 :
                    temp2 = 255
                if temp2 < 0 :
                    temp2 = abs(temp2)
                temp[i][j] = temp2
        cv2.imshow('Sobel Y',temp)


    def on_btn3_4_click(self):
        img = cv2.imread('.\\Data\\Q3_Image\\House.jpg',cv2.IMREAD_GRAYSCALE)
        gaf= [[0.045, 0.122, 0.045], [0.122, 0.332,0.122], [0.045, 0.122, 0.045]]
        solx= [[-1, 0, 1], [-2, 0,2], [-1, 0, 1]]
        soly= [[1,2,1], [0, 0,0], [-1, -2, -1]]

        img2 = img.copy()
        tempimg = img.copy()

        #x 存在img2
        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                img2[i][j]=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
        img = img2.copy()
        gaf = solx

        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                temp=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
                if temp > 255 :
                    temp = 255
                if temp < 0 :
                    temp = abs(temp)
                img2[i][j] = temp

        #y
        gaf= [[0.045, 0.122, 0.045], [0.122, 0.332,0.122], [0.045, 0.122, 0.045]]

        temp = tempimg.copy()

        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                temp[i][j]=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
        img = temp.copy()
        gaf = soly
        ans = temp.copy()
        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                temp2=gaf[0][0]*img[i-1][j-1]+gaf[0][1]*img[i-1][j]+gaf[0][2]*img[i-1][j+1]+gaf[1][0]*img[i][j-1]+gaf[1][1]*img[i][j]+gaf[1][2]*img[i][j+1]+gaf[2][0]*img[i+1][j-1]+gaf[2][1]*img[i+1][j]+gaf[2][2]*img[i+1][j+1]
                if temp2 > 255 :
                    temp2 = 255
                if temp2 < 0 :
                    temp2 = abs(temp2)
                temp[i][j] = temp2

        for i in range(1,img.shape[0]-2):
            for j in range(1,img.shape[1]-2):
                ans[i][j]= np.sqrt((img2[i][j]**2)+(temp[i][j]**2))

        cv2.imshow('magnitube',ans)


    def on_btn4_1_click(self):
        img = cv2.imread('.\\Data\\Q4_Image\\SQUARE-01.png')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        
        cv2.imshow('Resize', img)
    

    def on_btn4_2_click(self):
        img = cv2.imread('.\\Data\\Q4_Image\\SQUARE-01.png')
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)

        translation_matrix = np.float32([ [1,0,0], [0,1,60] ])
        trans_img = cv2.warpAffine(img, translation_matrix, (400, 300))

        cv2.imshow('trans',trans_img)


    def on_btn4_3_click(self):
        img = cv2.imread('.\\Data\\Q4_Image\\SQUARE-01.png')
        img = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        (h,w,d) = img.shape
        center = (w//2, h//2)

        M = cv2.getRotationMatrix2D(center, 10, 0.5)
        rotate_img = cv2.warpAffine(img, M, (400,300))

        translation_matrix = np.float32([ [1,0,0], [0,1,60] ])
        image = cv2.warpAffine(rotate_img, translation_matrix, (400, 300))

        cv2.imshow('rotate',image)


    def on_btn4_4_click(self):
        img = cv2.imread('.\\Data\\Q4_Image\\SQUARE-01.png')
        img = cv2.resize(img,(0,0),fx=0.5,fy=0.5,interpolation=cv2.INTER_AREA)
        (h,w,d) = img.shape
        center = (w//2, h//2)

        M = cv2.getRotationMatrix2D(center, 10, 0.5)
        rotate_img = cv2.warpAffine(img, M, (400,300))

        translation_matrix = np.float32([ [1,0,0], [0,1,60] ])
        image = cv2.warpAffine(rotate_img, translation_matrix, (400, 300))

        old_location = np.float32([[50,50],[200,50],[50,200]])
        new_location = np.float32([[10,100],[200,50],[100,250]])

        M_shearing = cv2.getAffineTransform(old_location,new_location)
        shearing_img = cv2.warpAffine(image, M_shearing, (400, 300))
        cv2.imshow('shearing',shearing_img)




    def click_on_image(window, event,x,y,flags,param):
        if event == cv2.EVENT_LBUTTONDOWN:
            global point_matrix
            global point_index
            point_matrix[point_index][0] = x
            point_matrix[point_index][1] = y
            point_index = point_index + 1


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())