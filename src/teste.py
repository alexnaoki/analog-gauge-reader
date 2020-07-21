import cv2 as cv
import numpy as np


# class teste:
#     def __init__(self, file):
#         self.file = file
#
#         self.rect_pts_BEFORE = np.float32([[834, 482], [940, 478], [924, 402], [834, 396]])
#         # self.rect_pts_AFTER = np.float32([[0,400],[400,400],[400,0],[0,0]])
#         self.rect_pts_AFTER = np.float32([[0,140],[140,140],[140,0],[0,0]])
#
#         video = cv2.VideoCapture(self.file)
#
#         # success, img = self.video.read()
#
#         for n_frame in range(0, 125000, 12500):
#             print(n_frame)
#             video.set(cv2.CAP_PROP_POS_FRAMES, n_frame)
#             _, img = video.read( )
#
#             img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
#             matrix = cv2.getPerspectiveTransform(self.rect_pts_BEFORE, self.rect_pts_AFTER)
#             result = cv2.warpPerspective(img_gray, matrix, (140, 140))
#
#             cv2.imshow('teste', result)
#             cv2.waitKey()





# a = teste(file=r'G:\Meu Drive\Colab Notebooks\opencv_data\ch02_20181221230757.mp4')


file = r'G:\Meu Drive\Colab Notebooks\opencv_data\ch02_20181227111237.mp4'
file_teste = r'C:\Users\User\Desktop\teste_opencv\ch01_test.mp4'
#
rect_pts_BEFORE = np.float32([[834, 482], [940, 478], [924, 402], [834, 396]])
# self.rect_pts_AFTER = np.float32([[0,400],[400,400],[400,0],[0,0]])
rect_pts_AFTER = np.float32([[0,140],[140,140],[140,0],[0,0]])

video = cv.VideoCapture(file_teste)
success, img = video.read()

print(video.get(cv.CAP_PROP_FRAME_COUNT))
# l
for n_frame in range(0, 10000, 100):
    print(n_frame)
    video.set(1, n_frame )
    _, img = video.read()

    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # matrix = cv.getPerspectiveTransform(rect_pts_BEFORE, rect_pts_AFTER)
    # result = cv.warpPerspective(img, matrix, (140, 140))

    cv.imshow('teste', img)
    cv.waitKey(-1)
# cap = cv2.VideoCapture(file)
# length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
#
# def onChange(trackbarValue):
#     cap.set(cv2.CAP_PROP_POS_FRAMES,trackbarValue)
#     err,img = cap.read()
#     cv2.imshow("mywindow", img)
#     pass
#
# cv2.namedWindow('mywindow')
# cv2.createTrackbar( 'start', 'mywindow', 0, length, onChange )
# cv2.createTrackbar( 'end'  , 'mywindow', 100, length, onChange )
#
# onChange(0)
# cv2.waitKey()
#
# start = cv2.getTrackbarPos('start','mywindow')
# end   = cv2.getTrackbarPos('end','mywindow')
# if start >= end:
#     raise Exception("start must be less than end")
#
# cap.set(cv2.CAP_PROP_POS_FRAMES,start)
# while cap.isOpened():
#     err,img = cap.read()
#     if cap.get(cv2.CAP_PROP_POS_FRAMES) >= end:
#         break
#     cv2.imshow("mywindow", img)
#     k = cv2.waitKey(10) & 0xff
#     if k==27:
#         break
