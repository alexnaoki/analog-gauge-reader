import cv2 as cv
import numpy as np
import pathlib

# class analog_gauge:
#     def __init__(self):
#         img_teste = pathlib.Path(r'G:\Meu Drive\Colab Notebooks\opencv_data\Resultado\img1.jpg')
#
#     def

img_teste = r'C:\Users\User\Desktop\teste_opencv\img1.jpg'
img = cv.imread(img_teste, cv.IMREAD_GRAYSCALE)

refPt = []
dpsPt = np.float32([[0,400],[400,400],[400,0],[0,0]])
print('Click duplo para selecionar ponto')
print('1: Bottom-Left\n2: Buttom-Right\n3: Top-Right\n4: Top-Left')
# mouse callback function
def draw_circle(event,x,y,flags,param):
    if event == cv.EVENT_LBUTTONDBLCLK:
        print(x,y)
        refPt.append([x, y])
        cv.circle(img,(x,y),5,(0,255,0),-1)
# Create a black image, a window and bind the function to window
# img = np.zeros((512,512,3), np.uint8)
cv.namedWindow('image')
cv.setMouseCallback('image',draw_circle)
while(1):
    if len(refPt) == 4:
        print(refPt)
        refPt_np = np.float32(refPt)
        matrix = cv.getPerspectiveTransform(refPt_np, dpsPt)
        result = cv.warpPerspective(img, matrix, (400,400))
        cv.imshow('teste',result)
        # print(result)
    cv.imshow('image',img)
    if cv.waitKey(20) & 0xFF == 27:
        break
cv.destroyAllWindows()
