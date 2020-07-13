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

        self.rect_pts_BEFORE = []
        self.rect_pts_AFTER = np.float32([[0,400],[400,400],[400,0],[0,0]])

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
        print(self.rect_pts_BEFORE)
        n = 10
        count = 0

        self.video_test = cv.VideoCapture(self.file)
        # success, image = self.video_test.read()
        while n>0:
            success, image = self.video_test.read()

            print(n)
            image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

            try:
                matrix = cv.getPerspectiveTransform(self.rect_pts_BEFORE_np, self.rect_pts_AFTER)
                result = cv.warpPerspective(image_gray, matrix, (135, 135))

                width, height = result.shape[:2]

                mask = np.zeros_like(result)
                mask = cv.circle(mask, center=(int(height/2), int(width/2)), radius=200, color=(255,255,255), thickness=-1)
                clock = np.bitwise_and(result, mask)

                clock_eq = cv.equalizeHist(clock)

                min_treshold = 100
                max_treshold = 230
                _, clock_t = cv.threshold(clock_eq, min_treshold, max_treshold, cv.THRESH_BINARY_INV)

                clock_c, _ = cv.findContours(clock_t, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

                for contour in clock_c:
                    area = cv.contourArea(contour)
                    print(area)
                    M = cv.moments(contour)
                    teste = []
                    # print(area)

                    # if area > 10000:
                    #     cx = int(M['m10']/M['m00'])
                    #     cy = int(M['m01']/M['m00'])
                    #     cv.circle(clock_t, (cx, cy), 0, (0,255,0), 5)
                    #
                    # if (area<10000) and (area>2500):
                    #     cx = int(M['m10']/M['m00'])
                    #     cy = int(M['m01']/M['m00'])
                    #     cv.circle(clock_t, (cx, cy), 0, (0,255,0), 5)
            except:
                pass

            # seconds = 90
            # count += self.fps * seconds
            # print(count)
            # self.video_test.set(1, count)
            # break
            n -= 1


        # try:
        #     self.rect_pts_BEFORE_np = np.float32(self.rect_pts_BEFORE)
        #     matrix = cv.getPerspectiveTransform(self.rect_pts_BEFORE_np, self.rect_pts_AFTER)
        #     result = cv.warpPerspective(self.image_gray_copy, matrix, (400, 400))
        #
        #     width, height = result.shape[:2]
        #
        #     mask = np.zeros_like(result)
        #     mask = cv.circle(mask, center=(int(height/2), int(width/2)), radius=200, color=(255,255,255), thickness=-1)
        #     clock = np.bitwise_and(result, mask)
        #
        #     clock_eq = cv.equalizeHist(clock)
        #
        #     min_treshold = 100
        #     max_treshold = 230
        #     _, clock_t = cv.threshold(clock_eq, min_treshold, max_treshold, cv.THRESH_BINARY_INV)
        #
        #     clock_c, _ = cv.findContours(clock_t, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        #
        #     # print(clock_c)
        #     for contour in clock_c:
        #         area = cv.contourArea(contour)
        #         print(area)
        #         M = cv.moments(contour)
        #         teste = []
        #         # print(area)
        #
        #         if area > 10000:
        #             cx = int(M['m10']/M['m00'])
        #             cy = int(M['m01']/M['m00'])
        #             cv.circle(clock_t, (cx, cy), 0, (0,255,0), 5)
        #
        #         if (area<10000) and (area>2500):
        #             cx = int(M['m10']/M['m00'])
        #             cy = int(M['m01']/M['m00'])
        #             cv.circle(clock_t, (cx, cy), 0, (0,255,0), 5)


        # cv.imshow('teste', clock_t)
        # cv.waitKey(-1)
        # except:
        #     print('Erro')

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

file = r'G:\Meu Drive\Colab Notebooks\opencv_data\ch01_20181210175338.mp4'
a = analog_gauge(file=file)
a.locate_gauge()
a.head_test()
