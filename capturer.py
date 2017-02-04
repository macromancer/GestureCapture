'''
https://gist.github.com/tedmiston/6060034
http://docs.opencv.org/3.1.0/dc/da5/tutorial_py_drawing_functions.html

ESC: Exit
Space: Capture Now
S: start/stop Auto Capture (Every 3 sec)
'''
import numpy as np
import cv2
import random as rand
import time
from datetime import datetime

def show_webcam(pose_name, dir='./'):
    cam = cv2.VideoCapture(0)

    ret_val, ori_img = cam.read()
    img_size = ori_img.shape[:2]

    print('img_size: height=%d, width=%d)' % img_size)

    center, radius = get_circle(img_size)

    capture_now = False
    auto_capture = False
    period = 3
    start_time = time.time()
    msg = ""

    while True:
        ret_val, ori_img = cam.read()
        img_size = ori_img.shape[:2]

        if capture_now or (auto_capture and period <= time.time() - start_time):
            file_name = '%s_%d_%d_%s.png' % (pose_name, center[0], center[1], datetime.now().strftime("%Y%m%d_%H%M%S"))
            cv2.imwrite(dir + file_name, ori_img)
            msg = '%s is saved.' % file_name
            capture_now = False

            center, radius = get_circle(img_size)
            start_time = time.time()

        else:
            display_img = np.copy(ori_img)
            #circle_img = np.zeros(ori_img.shape, np.uint8)
            color1 = (128,128,128)
            color2 = (200,200,200)

            cv2.circle(display_img, center, radius, color1, 4)

            start_deg = 270
            if auto_capture:
                add_deg = 360 * (time.time() - start_time) / period
            else:
                add_deg = 0

            cv2.ellipse(display_img, center, (radius+5, radius+5), 0, start_deg, start_deg + add_deg, color2, 4)

            cv2.putText(display_img, msg, (10,10), cv2.FONT_HERSHEY_COMPLEX, 0.5, color2)

            cv2.imshow('my webcam', display_img)

        key = cv2.waitKey(1)

        if key == 27: # esc
            break
        elif key == 32: # space
            capture_now = True
        elif chr(key & 255) == 's':
            auto_capture = not auto_capture

            if auto_capture:
                print('start auto capture')
                center, radius = get_circle(img_size)
                start_time = time.time()
            else:
                print('stop auto capture')

    cv2.destroyAllWindows()


def get_circle(img_size):
    center = (rand.randrange(0, img_size[1]+1), rand.randrange(0, img_size[0]+1))
    radius = rand.randrange(10,20)

    print('(%d,%d), %d' % (center[0], center[1], radius))
    return center, radius


# def main():
#     show_webcam(mirror=True)

if __name__ == '__main__':
    show_webcam('point')
    #main()
