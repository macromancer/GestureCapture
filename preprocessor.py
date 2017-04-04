import os
import sys
import numpy as np
import cv2
import random as rand
import time
from datetime import datetime


def chroma_key(source, dest):
    if not os.path.exists(source):
        raise IOError('Error: %s does not exists' % source)

    if not os.path.exists(dest):
        print('%s does not exist. mkdir %s' % (dest, dest))
        os.makedirs(dest)

    if not os.path.isdir(dest):
        print('Warning: destination path [%s] is a file. file name will be ignored.' % dest)
        dest = os.path.dirname(dest)


    if os.path.isdir(source):
        src_files = os.listdir(source)
        for filename in src_files:
            src_file = os.path.join(source, filename)
            chroma_key(src_file, dest)

        return

    print(os.getcwd())
    print(source)

    img = cv2.imread(source)

    #img2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bg_color = get_bgcolor(img_hsv)

    print("bgcolor", bg_color)

    bg = np.zeros_like(img).astype(np.int)
    bg[:,:,0] = bg_color[0]
    bg[:,:,1] = bg_color[1]
    bg[:,:,2] = bg_color[2]

    img_dist = np.abs(img_hsv.astype(np.int)[:,:,0] - bg[:,:,0])
    max_dist = np.max(img_dist)
    img_dist = img_dist * 255 / max_dist

    thre = 100

    a = np.zeros_like(b)
    a[img_dist > thre] = 255

    img2 = cv2.merge((b,g,r,a))

    filename = os.path.basename(source)
    filename = os.path.splitext(filename)[0]
    dest_file = os.path.join(dest, filename + '.png')
    cv2.imwrite(dest_file, img2)


def distance(color1, color2):
    return np.linalg.norm(color1 - color2)


def get_bgcolor(img):
    colors = np.array([img[0,0,:], img[0,-1,:], img[-1,0,:], img[-1,-1,:], img[-1, 200]])
    colors = colors.astype(int)

    num = len(colors)
    dist = np.zeros(num)
    for i in range(num):
        for j in range(num):
            dist[i] += distance(colors[i,:], colors[j,:])
            # print(distance(colors[i,:], colors[j,:]))

    colors = np.delete(colors, np.argmax(dist), 0)

    bg_color = np.mean(colors, axis=0).astype(np.uint8)
    return bg_color


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('python capture.py [function] [options]')
        exit()

    if sys.argv[1] == "chroma_key":
        if len(sys.argv) < 4:
            print('python capture.py chroma_key [source path] [destination dir path]')
            exit()

        chroma_key(sys.argv[2], sys.argv[3])
    else:
        print('invalid function [%s]' % sys.argv[1])
