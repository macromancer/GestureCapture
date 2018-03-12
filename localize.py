# -*- coding: utf-8 -*-

import numpy as np
import os
import random
import cv2
import sys
import re
import datetime
import cv2
import time

from keras.callbacks import ModelCheckpoint
from keras.models import Sequential
from keras.models import model_from_yaml
from keras.layers import Activation
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.utils import np_utils
from keras import metrics
from keras import backend as K
# K.set_image_dim_ordering('th')


def load_data(src, count=100, gesture_str_list=None, size=(256, 256)):

    if not os.path.exists(src):
        raise IOError('Error: %s does not exists'% src)

    list_file = os.listdir(src)

    data = np.zeros([count, size[0], size[1], 3], dtype=float)
    gestures_str = [None] * count
    centers = np.zeros([count, 2], dtype=float)

    for i in range(count):
        file_img = random.choice(list_file)

        label = parse_name(file_img)

        img = cv2.imread(src +'/'+ file_img)

        gestures_str[i] = label['gesture']
        centers[i, :] = np.array([float(label['center_x']), float(label['center_y'])])

        data[i, :, :, :] = img
        # labels[i] = (gesture, center)

    if gesture_str_list is None:
        gesture_str_list = list(set(gestures_str))

    gestures = np.zeros(count, dtype=int)
    for i in range(count):
        gestures[i] = gesture_str_list.index(gestures_str[i])

    return data, gestures, centers, gesture_str_list


def parse_name(file_name):
    filename_pattern = r'^(?P<gesture>\D+)_'\
                       r'(?P<hand>[LR]{1})_'\
                       r'(?P<rand_code>[A-Z0-9]{8})_'\
                       r'(?P<center_x>[\-\d\.]+)_(?P<center_y>[\-\d\.]+)'\
                       r'\.(?P<ext>\w+)'

    result = re.match(filename_pattern, file_name)

    return result.groupdict()


def learn_gesture_simple():
    N_train = 16
    N_test = 1000

    seed = 7
    np.random.seed(seed)

    X_train, y_train, _, labels = load_data('/media/kikim/Data/data/Gestures/img/20170524_train', count=N_train)
    X_train = X_train / 255.0

    # cv2.imshow(labels[y_train[0]], X_train[0])
    # key = cv2.waitKey(0)
    #
    # cv2.imshow(labels[y_train[1]], X_train[1])
    # key = cv2.waitKey(0)

    X_test, y_test, _, labels = load_data('/media/kikim/Data/data/Gestures/img/20170524_test', count=N_test,
                                          gesture_str_list=labels)
    X_test = X_test / 255.0

    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]

    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(256, 256, 3), data_format="channels_last",
                     padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.2))

    model.add(Conv2D(32, (3, 3), data_format="channels_last",
                     padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

    model.add(Flatten())
    model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # model.add(Conv2D(32, (3, 3), input_shape=(512, 512, 3), padding='same', activation='relu',
    #                  kernel_constraint=maxnorm(3), data_format="channels_last"))
    # model.add(Dropout(0.2))
    # model.add(Conv2D(32, (3, 3), activation='relu', padding='same',
    #                  kernel_constraint=maxnorm(3)))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Flatten())
    # model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(Dropout(0.5))
    # model.add(Dense(num_classes, activation='softmax'))

    epochs = 25
    lrate = 0.001
    decay = lrate / epochs
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    # Fit the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def learn():
    # N_train = 1024
    N_train = 8192
    # N_train = 16
    N_test = 1000

    seed = 7
    np.random.seed(seed)

    model = Sequential()
    model.add(Conv2D(32, (5, 5), input_shape=(256, 256, 3), data_format="channels_last",
                     padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.2))

    model.add(Conv2D(128, (3, 3), data_format="channels_last",
                     padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    # model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), data_format="channels_last",
                     padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))

    model.add(Conv2D(64, (3, 3), data_format="channels_last",
                     padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    # model.add(Dropout(0.2))

    # model.add(Conv2D(32, (3, 3), data_format="channels_last",
    #                  padding='same', activation='relu', kernel_constraint=maxnorm(3)))
    # model.add(MaxPooling2D(pool_size=(2, 2), data_format='channels_last'))
    # # model.add(Dropout(0.2))

    model.add(Flatten())
    # model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(196, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(64, activation='relu', kernel_constraint=maxnorm(3)))
    model.add(Dense(2, kernel_initializer='normal'))

    epochs = 10
    lrate = 0.001
    decay = 0.0
    # decay = lrate / epochs
    # sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    print(model.summary())

    # serialize model to YAML
    timestamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    model_yaml = model.to_yaml()
    with open("/media/kikim/Data/data/models/localize-model-%s.yaml" % timestamp, "w") as yaml_file:
        yaml_file.write(model_yaml)

    # define checkpoint
    weight_file = '/media/kikim/Data/data/models/localize-weights-%s-{epoch:03d}-{loss:.4f}.hdf5' % timestamp
    checkpoint = ModelCheckpoint(weight_file, monitor='loss', verbose=1, save_best_only=True, mode='min')
    callback_list = [checkpoint]

    for i in range(5):
        print('################# Iteration: %d' % i)

        X_train, _, y_train, labels = load_data('/media/kikim/Data/data/Gestures/img/20170813_train1', count=N_train)
        X_train = X_train / 255.0

        X_test, _, y_test, _ = load_data('/media/kikim/Data/data/Gestures/img/20170813_test', count=N_test,
                                         gesture_str_list=labels)
        # X_test, _, y_test, _ = load_data('/media/kikim/Data/data/Gestures/img/20170813_train', count=N_test,
        #                                  gesture_str_list=labels)
        X_test = X_test / 255.0

        y_train = y_train / 256.0 - 0.5
        y_test = y_test / 256.0 - 0.5

        y_train *= 2
        y_test *= 2

        # Fit the model
        model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
        # model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32, callbacks=callback_list)

        model.save_weights('/media/kikim/Data/data/models/localize-weights-%s.hdf5' % timestamp)

        # K.set_value(sgd.lr, lrate / (epochs * (i + 1)))

    # Final evaluation of the model
    scores = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy: %.2f%%" % (scores[1] * 100))


def test(model_file_path, weight_file_path):

    loaded_model = load_model(model_file_path, weight_file_path)
    loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    loaded_model.summary()

    X_train, _, y_train, labels = load_data('/media/kikim/Data/data/Gestures/img/20170813_train1', count=1000)
    X_train = X_train / 255.0

    X_test, _, y_test, _ = load_data('/media/kikim/Data/data/Gestures/img/20170813_test1', count=100,
                                     gesture_str_list=labels)

    print(X_train.shape)
    print(X_test.shape)
    print(np.min(y_test[:, 0]), np.max(y_test[:, 0]))
    print(np.min(y_test[:, 1]), np.max(y_test[:, 1]))

    X_test = X_test / 255.0

    y_train = y_train / 256.0 - 0.5
    y_test = y_test / 256.0 - 0.5

    print(np.min(y_test[:, 0]), np.max(y_test[:, 0]))
    print(np.min(y_test[:, 1]), np.max(y_test[:, 1]))

    p = loaded_model.predict(X_train, verbose=0)
    print(np.median(np.abs(p - y_train)))

    p = loaded_model.predict(X_test, verbose=0)
    abs_p = np.abs(p - y_test)
    print(np.median(abs_p))
    print(np.percentile(abs_p, 80))
    print(np.percentile(abs_p, 90))


def load_model(model_file_path, weight_file_path):
    model_file = open(model_file_path, 'r')
    loaded_model_yaml = model_file.read()
    model_file.close()
    model = model_from_yaml(loaded_model_yaml)

    model.load_weights(weight_file_path)

    model.summary()

    return model


def play(model_file_path, weight_file_path):

    loaded_model = load_model(model_file_path, weight_file_path)
    loaded_model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mae'])
    loaded_model.summary()

    cam = cv2.VideoCapture(0)

    ret_val, ori_img = cam.read()
    img_size = ori_img.shape[:2]

    input_size = (256, 256)


    # N_test = 10
    # X_test, _, y_test, _ = load_data('/media/kikim/Data/data/Gestures/img/20170605_test', count=N_test)
    # X_test = X_test / 255.0
    #
    # p = loaded_model.predict(X_test[:1,:,:,:], verbose=0)
    #
    # p = p + 0.5
    #
    # p = (int(p[0][0] * input_size[0]), int(p[0][1] * input_size[1]))
    # print(p)
    #
    # cv2.circle(X_test[0,:,:,:], p, 10, (0,0,255))
    #
    # while True:
    #     cv2.imshow('aaa', X_test[0,:,:,:])
    #     key = cv2.waitKey(1)
    #
    #     if key == 27:  # esc
    #         break


    while True:
        ret_val, ori_img = cam.read()
        img_size = ori_img.shape[:2]

        # ori_img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)

        x = cv2.resize(ori_img, input_size)
        x = x.reshape((1, input_size[0], input_size[1], 3))
        x = x / 255.0

        # X_test, _, y_test, _ = load_data('/media/kikim/Data/data/Gestures/img/20170813_test1', count=10)
        # x = X_test
        # x = x / 255.0

        p = loaded_model.predict(x, verbose=0)

        # p = p + 0.5

        p = (p + 1) / 2

        # p = (p[0] * img_size[0], p[1] * img_size[1])
        p = (int(p[0][0] * input_size[0]), int(p[0][1] * input_size[1]))

        print(p)

        cv2.circle(x[0,:,:,:], p, 10, (0,255,0))

        cv2.imshow('my webcam', x[0,:,:,:])

        # time.sleep(.5)
        key = cv2.waitKey(1)

        if key == 27:  # esc
            break

    cv2.destroyAllWindows()


if __name__ =='__main__':
    if sys.argv[1] == 'test_load_data':
        data, gestures, centers, _ = load_data('/media/kikim/Data/data/Gestures/img/20170813_train1', count=16)
        # data, gestures, centers = load_data('/media/kikim/Data/data/Gestures/img/20170502', count=16)

        print(data.shape)
        print(len(gestures))
        print(gestures[0])
        print(len(centers))
        print(centers[0])

        cv2.circle(data[0], (int(centers[0][0]), int(centers[0][1])), 10, (128, 0, 0), 3)
        cv2.imshow('demo_load_data', (data[0]))
        key = cv2.waitKey(0)

    elif sys.argv[1] == 'learn':
        # learn_gesture_simple()
        learn()
    elif sys.argv[1] == 'play':
        # play('/media/kikim/Data/data/models/localize-model-20170607083923.yaml',
        #      '/media/kikim/Data/data/models/localize-weights-20170607083923.hdf5')

        # test('/media/kikim/Data/data/models/localize-model-20170607083923.yaml',
        #      '/media/kikim/Data/data/models/localize-weights-20170607083923.hdf5')

        # test('/media/kikim/Data/data/models/localize-model-20180311113311.yaml',
        #      '/media/kikim/Data/data/models/localize-weights-20180311113311-009-0.0003.hdf5')

        # play('/media/kikim/Data/data/models/localize-model-20180311113311.yaml',
        #      '/media/kikim/Data/data/models/localize-weights-20180311113311-009-0.0003.hdf5')

        # play('/media/kikim/Data/data/models/localize-model-20180311142224.yaml',
        #      '/media/kikim/Data/data/models/localize-weights-20180311142224.hdf5')


        play('/media/kikim/Data/data/models/localize-model-20180312082634.yaml',
             '/media/kikim/Data/data/models/localize-weights-20180312082634.hdf5')
