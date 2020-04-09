import os
import cv2
import random
import argparse
import shutil
import numpy as np


class DataGenerator(object):

    @staticmethod
    def gen_toyset(data_dir):
        if os.path.exists(data_dir):
            return

        os.makedirs(data_dir)
        fw = open(os.path.join(data_dir, 'train_label.txt'), 'w')
        for i in range(50):
            for j in range(50):
                img = np.random.randint(i * 2, (i + 1) * 2, size=(224, 224, 3)).astype(np.uint8)
                img_path = os.path.abspath(os.path.join(data_dir, '{}_{}.jpg'.format(i, j)))
                cv2.imwrite(img_path, img)
                fw.write('{} {} {}\n'.format(img_path, i, i // 10))

        fw.close()

        fw = open(os.path.join(data_dir, 'val_label.txt'), 'w')
        for i in range(50):
            for j in range(5):
                img = np.random.randint(i * 2, (i + 1) * 2, size=(224, 224, 3)).astype(np.uint8)
                img_path = os.path.abspath(os.path.join(data_dir, 'test_{}_{}.jpg'.format(i, j)))
                cv2.imwrite(img_path, img)
                fw.write('{} {} {}\n'.format(img_path, i, i // 10))

        fw.close()

    @staticmethod
    def is_img(img_name):
        IMG_EXTENSIONS = [
            '.jpg', '.JPG', '.jpeg', '.JPEG',
            '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
            ]
        return any(img_name.endswith(extension) for extension in IMG_EXTENSIONS)

