import cv2 as cv
import numpy as np

class analog_gauge:
    def __init__(self, file):
        self.file = file
        self.video = cv.VideoCapture(file)
        self.fps = self.video.get(cv.CAP_PROP_FPS)
        print('FPS:', self.fps)

        while True:
            _, self.image = self.video.read()
            self.image_gray = cv.cvtColor(self.image, cv.COLOR_BGR2GRAY)
            self.image_gray_copy = self.image_gray.copy()
            break
        cv.destroyAllWindows()
        self.video.release()

        self.rect_pts_BEFORE = []
        # self.rect_pts_AFTER = np.float32([[0,400],[400,400],[400,0],[0,0]])
        self.rect_pts_AFTER = np.float32([[0,135],[135,135],[135,0],[0,0]])

    def locate_gauge(self):
        print('asdfasfdsa')
        print(np.shape(self.image_gray))
        cv.namedWindow('imag')
        cv.setMouseCallback('imag', self.locate_pts)

        cv.imshow('imag',self.image_gray)
        cv.waitKey(-1)
        cv.destroyAllWindows()

    def head_test(self):
        self.rect_pts_BEFORE_np = np.float32(self.rect_pts_BEFORE)
        self.rect_pts_BEFORE_np = np.float32([[628, 665], [752, 660], [750, 574], [626, 573]])
        self.rect_pts_BEFORE_np = np.float32([[625, 670], [749, 658], [752, 570], [625, 574]])
        print(self.rect_pts_BEFORE)
        # n = 20
        count = 20
        seconds = 30
        video_test = cv.VideoCapture(self.file)
        # success, image = self.video_test.read()
        # while count < n:
        n_frame = 0
        # while video_test.isOpened():
        # for frame in range(0, 125000, 12500):
        while n_frame <= count*self.fps*seconds:
            video_test.set(1, n_frame)
            print(n_frame)
            process = True

            # print('N FRAME',n_frame)
            # frameid = self.video_test.set(1,n_frame)
            success, image = video_test.read()

            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            matrix = cv.getPerspectiveTransform(self.rect_pts_BEFORE_np, self.rect_pts_AFTER)
            result = cv.warpPerspective(image_gray, matrix, (140, 140))

            if process == True:
                try:
                    width, height = result.shape[:2]

                    mask = np.zeros_like(result)
                    mask = cv.circle(mask, center=(int(height/2), int(width/2)), radius=72, color=(255,255,255), thickness=-1)
                    clock = np.bitwise_and(result, mask)

                    clock_eq = cv.equalizeHist(clock)

                    min_treshold = 70
                    max_treshold = 210
                    _, clock_t = cv.threshold(clock_eq, min_treshold, max_treshold, cv.THRESH_BINARY_INV)

                    clock_c, _ = cv.findContours(clock_t, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
                    # clock_c, _ = cv.findContours(clock_e, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE)
                    for contour in clock_c:
                        area = cv.contourArea(contour)
                        print('area',area)
                        M = cv.moments(contour)
                        teste = []
                        # print(area)
                    #
                        if (area > 3000) and (area<4100):
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                            # print(cx, cy)
                            # cv.circle(clock_t, (cx, cy), 5, (0,0,0), -1)
                            index_far, far = self.most_distance_pts_from_contour(centroid=(70,70), pts=np.squeeze(contour))
                            x_angle = contour[index_far][0,0] - 70
                            y_angle = 70 - contour[index_far][0, 1]
                    #
                            # cv.circle(clock_t, (contour[index_far][0,0], contour[index_far][0,1]), 10, (255,255,255),10)
                    #
                        elif area > 4100:
                            cx = int(M['m10']/M['m00'])
                            cy = int(M['m01']/M['m00'])
                            # print(cx, cy)
                            # cv.circle(clock_eq, (cx, cy), 5, (255,0,0), -1)

                            hull = cv.convexHull(contour, returnPoints=False)
                            defects = cv.convexityDefects(contour, hull)

                            for i in range(defects.shape[0]):
                                s, e, f, d = defects[i, 0]
                                teste.append(contour[f])

                            index_far, far = self.most_distance_pts_from_contour(centroid=(70,70), pts=np.squeeze(teste))

                            x_angle = teste[index_far][0,0] - 70
                            y_angle = 70 - teste[index_far][0,1]

                            # cv.circle(clock_eq, (teste[index_far][0,0], teste[index_far][0,1]), 10, (255,255,255), 10)
                        else:
                            pass

                    values = self.get_angle(x_angle=x_angle, y_angle=y_angle)
                    print(values)


                    cv.imshow('dfasdfasf', clock_t)
                    cv.imshow('eq', clock_eq)
                    cv.imshow('noeq', clock)
                    # cv.imshow('clahe', clock_clahe)
                    # cv.imshow('dfadfs',clock_eq)
                    cv.waitKey(-1)
                    # cv.destroyAllWindows()

                except:
                    frameid = video_test.set(1, n_frame)
                    print('erro')
                    video_test.release()
                    cv.waitKey(-1)

            elif process == False:
                cv.imshow('no modification', result)
                cv.waitKey(-1)

            n_frame += int(self.fps*seconds)

    def locate_pts(self, event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDBLCLK:
            print(x, y)
            self.rect_pts_BEFORE.append([x, y])
            cv.circle(self.image_gray, (x, y), 5, (255,0,0), -1)
            # cv.polylines(self.image_gray, np.int32(self.rect_pts_BEFORE), True, (0,255,0))
            print(self.rect_pts_BEFORE)

            cv.imshow('imag', self.image_gray)

    def most_distance_pts_from_contour(self, centroid, pts):
        fartest = 0
        for i, point in enumerate(pts):
            dist = ((centroid[0] - point[0])**2 + (centroid[1] - point[1])**2)**(1/2)
            # print(point, dist)
            if i == 0:
                fartest = dist
                index = i
            elif fartest < dist:
                fartest = dist
                index = i
        return index, fartest

    def get_angle(self, x_angle, y_angle):
        res = np.arctan(np.divide(float(y_angle), float(x_angle)))
        res = np.rad2deg(res)
        print(x_angle, y_angle)
        print((x_angle + 70)-70, (70-y_angle)-70)
                            # x_angle = contour[index_far][0,0] - 70
                            # y_angle = 70 - contour[index_far][0,1]
        if x_angle > 0 and y_angle > 0:  #in quadrant I 0 ~ 2,5
            print('quadrante 1')
            # final_angle = 270 - res
            final_angle = res
            new_value = 2.5 - 2.5/90*final_angle
        if x_angle < 0 and y_angle > 0:  #in quadrant II 7,5 ~ 10
            print('quadrante 2')
            final_angle = 180 + res
            new_value = 10 - 2.5/90*(final_angle-90)
        if x_angle < 0 and y_angle < 0:  #in quadrant III 5 ~ 7,5
            print('quadrante 3')
            # final_angle = 90 - res
            final_angle = 180 + res
            new_value = 7.5 - 2.5/90*(final_angle-180)
        if x_angle > 0 and y_angle < 0:  #in quadrant IV 2,5 ~ 5
            print('quadrante 4')

            final_angle = 360 + res
            new_value = 5 - 2.5/90*(final_angle-270)
        return res, final_angle, new_value

file = r'G:\Meu Drive\Colab Notebooks\opencv_data\ch02_20181221230757.mp4'
file_teste = r'C:\Users\User\Desktop\teste_opencv\ch01_test.mp4'

a = analog_gauge(file=file_teste)
a.locate_gauge()
a.head_test()
