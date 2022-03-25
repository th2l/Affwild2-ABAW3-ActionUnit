"""
Author: HuynhVanThong
Department of AI Convergence, Chonnam Natl. Univ.
"""
import os.path

import numpy as np
import pandas as pd

from tools.mtcnn import MTCNN
import torch
import cv2
import argparse
import matplotlib.pyplot as plt
from tqdm import tqdm


def create_expw(dataset_root='/mnt/Work/Dataset/ExpW/'):
    expw_label = pd.read_csv(os.path.join(dataset_root, 'label.lst'), header=None, sep=' ')
    os.makedirs(os.path.join(dataset_root, 'faces'), exist_ok=True)

    expw_2_affwild2 = {0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 0}
    expw_2_affwild2_data = []
    for index, row in tqdm(expw_label.iterrows(), total=expw_label.shape[0]):
        img_path = os.path.join(dataset_root, 'origin', row[0])
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        t, l, r, b = row[2:6]
        face_img = img[t:b, l:r]
        aff2_emotion = expw_2_affwild2[int(row[7])]

        file_name = '{}_{}.jpg'.format(row[0][:-4], int(row[1]))
        cv2.imwrite(os.path.join(dataset_root, 'faces/{}'.format(file_name)), face_img)
        expw_2_affwild2_data.append(['faces/{}'.format(file_name), aff2_emotion, int(row[1])])

    np.save('expw_data.npz', expw_2_affwild2_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Data preparation ABAW3 - CVPR 2022')
    parser.add_argument('--root_dir', type=str, default='/mnt/Work/Dataset/Affwild2_ABAW3/', help='Root data folder')
    parser.add_argument('--out_dir', type=str, default='/mnt/Work/Dataset/Affwild2_ABAW3/', help='Out data folder')
    args = parser.parse_args()

    create_expw()
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # mtcnn = MTCNN(image_size=112, margin=0, min_face_size=20, thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=False, device=device, align=True, to_tensor=False)
    #
    # img_path = '/home/hvthong/sXProject/Affwild2_ABAW3/tools/face_example.png'
    # img = cv2.imread(img_path)
    # face = mtcnn(img)
