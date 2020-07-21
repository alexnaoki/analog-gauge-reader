import cv2 as cv
import numpy as np
import datetime as dt

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, Toggle, Div
from bokeh.plotting import figure
from bokeh.layouts import row, column
from bokeh.palettes import RdBu4, Spectral4

class read_analog_clock:
    def __init__(self):
        print('entrou')

        self.file = r'C:\Users\User\Desktop\teste_opencv\ch01_test.mp4'


        self.source_01 = ColumnDataSource(dict(time=[], value=[], mean=[], diff=[], cumulative=[]))
        self.img_source_01 = ColumnDataSource(dict(img=[]))

        fig_01 = figure(plot_height=150, plot_width=500, tools='xpan,xwheel_zoom,xbox_zoom,reset')
        fig_01.circle(x='time', y='value', source=self.source_01, legend_label='OPENCV READS', color=Spectral4[2])
        fig_01.step(x='time', y='diff', source=self.source_01, color=Spectral4[1], line_width=2, legend_label='DIFFERENCE')
        fig_01.legend.location = 'bottom_left'

        # fig_01.yaxis.axis_label = 'Hydrometer Value'

        fig_03 = figure(plot_height=200, plot_width=500, tools='xpan,xwheel_zoom,xbox_zoom,reset')
        fig_03.line(x='time', y='cumulative', source=self.source_01, line_width=2, color=Spectral4[3], legend_label='TOTAL VOLUME (L)')
        fig_03.legend.location = 'bottom_right'

        # fig_03.yaxis.axis_label = 'Total Volume (L)'

        fig_04 = figure(plot_height=200, plot_width=500, tools='xpan,xwheel_zoom,xbox_zoom,reset', y_range=(0,1))
        fig_04.line(x='time', y='mean', source=self.source_01, line_width=2, color=Spectral4[0], legend_label='FLOW RATE (L/s)')

        # fig_04.yaxis.axis_label = 'Flow Rate (L/s)'

        fig_02 = figure(plot_width=650, plot_height=500,
                        x_axis_type=None, y_axis_type=None, tools='pan,wheel_zoom,', name='image')
        fig_02.image_rgba('img', x=0, y=0, dw=140, dh=140, source=self.img_source_01)

        fig_01.toolbar.autohide = True
        fig_02.toolbar.autohide = True
        fig_03.toolbar.autohide = True
        fig_04.toolbar.autohide = True

        fig_04.xaxis.axis_label = 'Time (s)'

        self.toggle = Toggle(label='Run/Stop', button_type='success')

        self.t = dt.datetime.now()
        self.time_video = 0

        self.cumulative = 0

        div01 = Div(text='''<b>@AlexNAKobayashi</b>''', style={'font-size':'150%'})


        self.video = cv.VideoCapture(self.file)
        self.n_frame = 0

        self.rect_pts_BEFORE_np = np.float32([[628, 665], [752, 660], [750, 574], [626, 573]])
        self.rect_pts_AFTER = np.float32([[0,140],[140,140],[140,0],[0,0]])
        self.matrix = cv.getPerspectiveTransform(self.rect_pts_BEFORE_np, self.rect_pts_AFTER)

        curdoc().add_root(column(row(column(fig_02, div01),column(fig_01, fig_03, fig_04)), self.toggle))
        curdoc().add_periodic_callback(self._calc_analog_clock, 1)

    def _calc_analog_clock(self):
        if self.toggle.active:

            self.seconds = 3
            self.fps = 25

            self.n_frame += self.seconds*self.fps

            self.video.set(1, self.n_frame)

            success, img = self.video.read()
            img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            result = cv.warpPerspective(img_gray, self.matrix, (140, 140))

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

                for contour in clock_c:
                    area = cv.contourArea(contour)
                    print('area',area)
                    M = cv.moments(contour)
                    teste = []

                    if (area > 3000) and (area<4100):
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        # print(cx, cy)
                        cv.circle(clock_t, (cx, cy), 5, (0,0,0), -1)
                        index_far, far = self.most_distance_pts_from_contour(centroid=(70,70), pts=np.squeeze(contour))
                        x_angle = contour[index_far][0,0] - 70
                        y_angle = 70 - contour[index_far][0, 1]
                        cv.circle(clock_t, (contour[index_far][0,0], contour[index_far][0,1]), 10, (255,255,255),10)

                    elif area > 4100:
                        cx = int(M['m10']/M['m00'])
                        cy = int(M['m01']/M['m00'])
                        # print(cx, cy)
                        cv.circle(clock_eq, (cx, cy), 5, (255,0,0), -1)

                        hull = cv.convexHull(contour, returnPoints=False)
                        defects = cv.convexityDefects(contour, hull)

                        for i in range(defects.shape[0]):
                            s, e, f, d = defects[i, 0]
                            teste.append(contour[f])

                        index_far, far = self.most_distance_pts_from_contour(centroid=(70,70), pts=np.squeeze(teste))

                        x_angle = teste[index_far][0,0] - 70
                        y_angle = 70 - teste[index_far][0,1]

                        cv.circle(clock_eq, (teste[index_far][0,0], teste[index_far][0,1]), 10, (255,255,255), 10)
                    else:
                        pass

                values = self.get_angle(x_angle=x_angle, y_angle=y_angle)
                print(values)

                all_values = self.source_01.data['value'] + [values[-1]]
                print(np.mean(all_values))

                if len(all_values)==1:
                    # print('SEM REF')
                    self.cumulative += values[-1]
                    diff = values[-1]

                    moving_mean = self.cumulative/self.seconds
                    # lastest_values = values[-1]
                else:
                    # print('MAIS DE UM')
                    diff = (values[-1] - self.source_01.data['value'][-1])
                    if diff >= 0:
                        print(diff)
                        if diff > 8:
                            self.cumulative += 0
                        else:
                            self.cumulative += diff
                    elif diff < 0:
                        self.cumulative += 0
                n = 5
                if len(all_values) == 1:
                    moving_mean = self.cumulative/self.seconds
                elif (len(all_values)>1) & (len(all_values)<=n-1):
                    moving_mean = (self.cumulative - self.source_01.data['cumulative'][-1])/(len(all_values)*self.seconds)
                else:
                    moving_mean = (self.cumulative - self.source_01.data['cumulative'][-2])/(n*self.seconds)

                self.time_video += self.seconds
                    # time=[(dt.datetime.now()-self.t).total_seconds()*1000],

                new_data = dict(
                    time=[self.time_video],
                    value=[values[-1]],
                    mean=[moving_mean],
                    diff=[diff],
                    cumulative=[self.cumulative]
                )


                img_frame = cv.cvtColor(img, cv.COLOR_BGR2RGBA)
                img_frame = img_frame.view(dtype="uint32").reshape(img.shape[:2])
                self.img_source_01.data = dict(img=[img_frame[::-5,::5]])

                self.source_01.stream(new_data, 100)

            except:
                print('erro')
                pass



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

        if x_angle > 0 and y_angle > 0:  #in quadrant I
            print('quadrante 1')
            # final_angle = 270 - res
            final_angle = res
            new_value = 2.5 - 2.5/90*final_angle
        if x_angle < 0 and y_angle > 0:  #in quadrant II
            print('quadrante 2')

            final_angle = 180 + res
            new_value = 10 - 2.5/90*(final_angle-90)
        if x_angle < 0 and y_angle < 0:  #in quadrant III
            print('quadrante 3')

            # final_angle = 90 - res
            final_angle = 180 + res
            new_value = 7.5 - 2.5/90*(final_angle-180)
        if x_angle > 0 and y_angle < 0:  #in quadrant IV
            print('quadrante 4')

            final_angle = 360 + res
            new_value = 5 - 2.5/90*(final_angle-270)
        return res, final_angle, new_value
        # video = cv.VideoCapture(self.file)



read_analog_clock()
