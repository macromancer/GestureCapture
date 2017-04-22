import os
import sys
import numpy as np
import cv2
import re
import random as rand
import time
from datetime import datetime


def chroma_key(src, dest):
    if not os.path.exists(src):
        raise IOError('Error: %s does not exists' % src)

    if not os.path.exists(dest):
        print('%s does not exist. mkdir %s' % (dest, dest))
        os.makedirs(dest)

    if not os.path.isdir(dest):
        print('Warning: destination path [%s] is a file. file name will be ignored.' % dest)
        dest = os.path.dirname(dest)

    if os.path.isdir(src):
        src_files = os.listdir(src)
        for filename in src_files:
            src_file = os.path.join(src, filename)
            chroma_key(src_file, dest)

        return

    print(os.getcwd())
    print(src)

    img = cv2.imread(src)

    b = img[:, :, 0]
    g = img[:, :, 1]
    r = img[:, :, 2]

    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    bg_color = get_bgcolor(img_hsv)

    print("bgcolor", bg_color)

    bg = np.zeros_like(img).astype(np.int)
    bg[:, :, 0] = bg_color[0]
    bg[:, :, 1] = bg_color[1]
    bg[:, :, 2] = bg_color[2]

    img_dist = np.abs(img_hsv.astype(np.int)[:, :, 0] - bg[:, :, 0])
    max_dist = np.max(img_dist)
    img_dist = img_dist * 255 / max_dist

    threshold = 100

    a = np.zeros_like(b)
    a[img_dist > threshold] = 255

    img2 = cv2.merge((b, g, r, a))

    [min_x, min_y, max_x, max_y] = get_bounding_rect(a)
    # img2 = cv2.rectangle(img2, (min_x, min_y), (max_x, max_y), (0, 0, 255, 255), 1)

    filename = os.path.basename(src)
    filename = os.path.splitext(filename)[0]
    bounding_rect_str = "_%d_%d_%d_%d" % (min_x, min_y, max_x, max_y)
    dest_file = os.path.join(dest, filename + bounding_rect_str + '.png')
    cv2.imwrite(dest_file, img2)


def distance(color1, color2):
    return np.linalg.norm(color1 - color2)


def get_bgcolor(img):
    colors = np.array([img[0, 0, :], img[0, -1, :], img[-1, 0, :], img[-1, -1, :], img[-1, 200]])
    colors = colors.astype(int)

    num = len(colors)
    dist = np.zeros(num)
    for i in range(num):
        for j in range(num):
            dist[i] += distance(colors[i, :], colors[j, :])
            # print(distance(colors[i,:], colors[j,:]))

    colors = np.delete(colors, np.argmax(dist), 0)

    bg_color = np.mean(colors, axis=0).astype(np.uint8)
    return bg_color


def get_bounding_rect(a):
    nonzero_x = np.nonzero(a.sum(axis=0))
    nonzero_y = np.nonzero(a.sum(axis=1))

    return np.min(nonzero_x), np.min(nonzero_y), np.max(nonzero_x) + 1, np.max(nonzero_y) + 1


def blend(src_fg, src_bg, dest, count=None):
    if not os.path.exists(src_fg):
        raise IOError('Error: %s does not exists' % src_fg)

    if not os.path.exists(src_bg):
        raise IOError('Error: %s does not exists' % src_bg)

    if not os.path.exists(dest):
        print('%s does not exist. mkdir %s' % (dest, dest))
        os.makedirs(dest)

    if not os.path.isdir(dest):
        print('Warning: destination path [%s] is a file. file name will be ignored.' % dest)
        dest = os.path.dirname(dest)


def blend_img_file(fg_file, bg_file, dest_dir, seq=None, ext='png'):
    if not os.path.exists(fg_file):
        raise IOError('Error: %s does not exists' % fg_file)

    if not os.path.exists(bg_file):
        raise IOError('Error: %s does not exists' % bg_file)

    if not os.path.exists(dest_dir):
        print('%s does not exist. mkdir %s' % (dest_dir, dest_dir))
        os.makedirs(dest_dir)

    if not os.path.isdir(dest_dir):
        print('Warning: destination path [%s] is a file. file name will be ignored.' % dest_dir)
        dest_dir = os.path.dirname(dest_dir)

    fg_img = cv2.imread(fg_file, cv2.IMREAD_UNCHANGED)
    print('fg_img.shape=', fg_img.shape)

    bg_img = cv2.imread(bg_file)
    blended_img = blend_img(fg_img, bg_img)

    filename = os.path.basename(fg_file)
    filename = os.path.splitext(filename)[0]

    seq_str = ''
    if seq is not None:
        seq_str = '_%d' % seq

    dest_file = os.path.join(dest_dir, filename + seq_str + '.' + ext)
    cv2.imwrite(dest_file, blended_img)


def blend_img(fg, bg):
    alpha = fg[:, :, 3:] / 255.  # height x width x 1
    img = np.multiply(np.tile(alpha, (1, 1, 3)), fg[:, :, :3]) + \
          np.multiply(np.tile(1.0 - alpha, (1, 1, 3)), bg[:, :, :3])

    return img


def parse_name(file_name):
    filename_pattern = r'^(?P<gesture>\D+)_(?P<center_x>\d+)_(?P<center_y>\d+)_(?P<center_r>\d+)_' \
                       r'(?P<date>\d+)_(?P<time>\d+)_' \
                       r'(?P<min_x>\d+)_(?P<min_y>\d+)_(?P<max_x>\d+)_(?P<max_y>\d+)' \
                       r'\.(?P<ext>\w+)'

    result = re.match(filename_pattern, file_name)

    return result.groupdict()


def transform(img, center_x, center_y, min_x, min_y, max_x, max_y, seed=None, delta=0.1):
    (height, width) = img.shape[:2]
    sigma = np.float32([width, height]) * delta

    # original reference points
    p = np.float32([[min_x, max_y],
                    [max_x, max_y],
                    [max_x, min_y]])

    # reference points after transformation
    np.random.seed(seed)
    q = np.random.normal(p, np.tile(sigma, (3, 1))).astype(np.dtype('float32'))

    if max_y == height:
        q[0][1] = max(height, q[0][1])
        q[1][1] = max(height, q[1][1])

    if max_x == width:
        q[1][0] = max(width, q[1][0])
        q[2][0] = max(width, q[2][0])

    M = cv2.getAffineTransform(p, q)
    trans_img = cv2.warpAffine(img, M, (width, height))

    trans_center = np.dot(M, np.transpose(np.array([center_x, center_y, 1]))).astype(int)

    return trans_img, trans_center


def mirror(img, center_x, center_y):
    width = img.shape[1]
    mirrored_img = cv2.flip(img, 1)
    mirrored_center = np.array([width - center_x, center_y])

    return mirrored_img, mirrored_center


def resize(img, width, height):
    resized_img = cv2.resize(img, (width, height))

    return resized_img


def crop_randomly(img, width=None, height=None, crop_ratio=1.0, aspect_ratio=None, mandatory_point=None):
    ori_height, ori_width = img.shape[:2]

    if crop_ratio > 1.0:
        raise ArithmeticError('crop_ratio should be less than 1.0, but %f' % crop_ratio)
        return

    if aspect_ratio <= 0:
        raise ArithmeticError('aspect_ratio should be greater than 0.0, but %f' % aspect_ratio)
        return

    if width is None:
        width = crop_ratio * ori_width

    if height is None:
        height = crop_ratio * ori_height

    if aspect_ratio is not None:
        if width > aspect_ratio * height:
            width = int(aspect_ratio * height)
        elif width < aspect_ratio * height:
            height = int(width / float(aspect_ratio))

    width = int(min(width, ori_width))
    height = int(min(height, ori_height))

    max_try = 100
    while True:
        min_x = np.random.randint(0, ori_width - width + 1)
        min_y = np.random.randint(0, ori_height - height + 1)

        max_x = min_x + width
        max_y = min_y + height

        if mandatory_point is None or (min_x <= mandatory_point[0] < max_x) and (min_y <= mandatory_point[1] < max_y):
            break

        max_try -= 1

        if max_try < 0:
            raise RuntimeError(
                'random cropping is failed. mandatory_point=(%d, %d)' % (mandatory_point[0], mandatory_point[1]))

    if mandatory_point is None:
        return img[min_y:max_y, min_x:max_x], np.array([min_x, min_y])


def demo_resize_bg(src, dest, width, height, rand_crop=None, transform_delta=None, seed=None):
    img = cv2.imread(src)

    if transform_delta is not None:
        img, _ = transform(img, 0, 0, 0, 0, img.shape[1], img.shape[0], delta=transform_delta)

    if rand_crop is not None:
        img, _ = crop_randomly(img, crop_ratio=rand_crop, aspect_ratio=width / float(height))

    img_resized = resize(img, width, height)

    cv2.imwrite(dest, img_resized)


def demo_resize_fg(src, dest, width, height, rand_crop=None, transform_delta=None, seed=None):
    img = cv2.imread(src)

    g = parse_name(os.path.basename(src))
    center = np.array([int(g['center_x']), int(g['center_y'])])

    if transform_delta is not None:
        img, _ = transform(img, 0, 0, 0, 0, img.shape[1], img.shape[0], delta=transform_delta)

    if rand_crop is not None:
        img, min_point = crop_randomly(img, crop_ratio=rand_crop, aspect_ratio=width / float(height),
                                       mandatory_point=center)

        center = center - min_point

    img_resized = resize(img, width, height)

    cv2.imwrite(dest, img_resized)


def demo_transform(src, dest, seed=None, delta=0.1):
    img = cv2.imread(src)

    g = parse_name(os.path.basename(src))

    trans_img, trans_center = transform(img,
                                        int(g['center_x']), int(g['center_y']),
                                        int(g['min_x']), int(g['min_y']),
                                        int(g['max_x']), int(g['max_y']),
                                        seed=seed, delta=delta)
    trans_img, trans_center = mirror(trans_img, trans_center[0], trans_center[1])
    trans_img = cv2.circle(trans_img, (trans_center[0], trans_center[1]), 50, (0, 0, 255), thickness=2)
    dest_file = dest + os.path.basename(src)
    cv2.imwrite(dest_file, trans_img)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('python capture.py [function] [options]')
        print('python capture.py chroma_key [source path] [destination dir path]')
        print('python capture.py blend [fg-image source path] [bg-image source path] [destination dir path]')
        print('python capture.py preprocess [transform.conf file path]')
        exit()

    if sys.argv[1] == "chroma_key":
        if len(sys.argv) < 4:
            print('python capture.py chroma_key [source path] [destination dir path]')
            print('ex) python capture.py chroma_key ./img/ori/ ./img/chroma/')
            exit()

        chroma_key(sys.argv[2], sys.argv[3])

    elif sys.argv[1] == "blend":
        if len(sys.argv) < 5:
            print('python capture.py blend [fg-image source path] [bg-image source path] [destination dir path]')
            print('python capture.py blend ' +
                  './imgs/chroma/Pointing_Thumb_Up_691_235_50_20170304_110354_649_35_1920_1080.png ' +
                  './imgs/bg/No_Gesture_45_115_50_20170305_120951.jpg ' + './imgs/modified/')
            exit()
        blend_img_file(sys.argv[2], sys.argv[3], sys.argv[4])
    elif sys.argv[1] == "parse":
        m = parse_name('Pointing_Thumb_Up_691_235_50_20170304_110354_649_35_1920_1080.png')
        print(m)
    elif sys.argv[1] == "demo_transform":
        demo_transform(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "demo_resize":
        demo_resize_bg('./imgs/bg/No_Gesture_182_553_50_20170305_114143.jpg', './imgs/modified/just_resized.png', 512,
                       512)
        demo_resize_bg('./imgs/bg/No_Gesture_182_553_50_20170305_114143.jpg', './imgs/modified/rand_crop_1.png', 512,
                       512, rand_crop=1.)
        demo_resize_bg('./imgs/bg/No_Gesture_182_553_50_20170305_114143.jpg', './imgs/modified/rand_crop_2.png', 512,
                       512, rand_crop=0.6)
    else:
        print('invalid function [%s]' % sys.argv[1])
