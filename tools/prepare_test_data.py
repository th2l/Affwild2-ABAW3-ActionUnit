import sys

import pandas as pd
import numpy as np
import pathlib
import argparse
from tqdm import tqdm
import cv2
import glob
import os

if __name__ == '__main__':
    root_video_path = '/home/hvthong/Documents/batch_1_2'
    dataset_path = '/home/hvthong/sXProject/Affwild2_ABAW3/dataset'

    for chal in [ 'EXPR',]: #'AU', 'VA',
        with open('{}/testset/{}_test_set_release.txt'.format(dataset_path, chal), 'r') as fd:
            list_test_video = fd.read().splitlines()
            chal_test_dataset = {}
            for vid in tqdm(list_test_video):
                if '_left' or '_right' in vid:
                    vid_name = vid.replace('_left', '').replace('_right', '') + '.mp4'
                else:
                    vid_name = vid + '.mp4'
                if not os.path.isfile('{}/{}'.format(root_video_path, vid_name)):
                    vid_name = vid_name.replace('.mp4', '.avi')
                    if not os.path.isfile('{}/{}'.format(root_video_path, vid_name)):
                        print('Could not find video with name: ', vid_name)

                cropped_aligned = glob.glob('{}/cropped_aligned/{}/*.jpg'.format(dataset_path, vid))
                cap = cv2.VideoCapture('{}/{}'.format(root_video_path, vid_name))
                # Count number of frame
                num_frames = 0
                while True:
                    ret, _ = cap.read()
                    if not ret:
                        break
                    num_frames += 1
                cap.release()

                cropped_aligned_largest = np.max([int(x.split('/')[-1].split('.')[0]) for x in cropped_aligned])
                list_files = np.array([None] * max(num_frames, cropped_aligned_largest))
                num_frames = len(list_files)
                for vid_frame in cropped_aligned:
                    list_files[int(vid_frame.split('/')[-1].split('.')[0]) - 1] = vid_frame.replace(
                        '{}/cropped_aligned/'.format(dataset_path), '')

                # print(vid)
                # Missing frame, copy existing frames to fill the empty
                st = 0
                while True:
                    while st < num_frames and list_files[st] is not None:
                        st += 1
                    if st == num_frames:
                        break
                    # list_files[st] is None
                    ed = st + 1
                    while ed < num_frames and list_files[ed] is None:
                        ed += 1

                    if st == 0:
                        list_files[st:ed] = list_files[ed]
                    elif ed == num_frames:
                        list_files[st:ed] = list_files[st-1]
                    else:
                        if ed - st > 2:
                            mid_point = (st+ed) // 2
                            list_files[st:mid_point] = list_files[st-1]
                            list_files[mid_point: ed] = list_files[ed]
                        else:
                            list_files[st:ed] = list_files[ed]

                    # ed == num_frames or list_files[ed] is not None, list_files[ed-1] is None
                    # Need to fill from st, ..., ed-1
                    # print(st, ed)
                    # after filled
                    st = ed

                if None in list_files:
                    print('Check again: ', vid)
                list_files = list_files.reshape(-1, 1)
                if chal == 'AU':
                    dummy_anno = -1* np.ones((len(list_files), 12), dtype=int)
                elif chal == 'VA':
                    dummy_anno = -5*np.ones((len(list_files), 2), dtype=float)
                elif chal == 'EXPR':
                    dummy_anno = -1*np.ones((len(list_files), 1), dtype=int)
                else:
                    print('Do not support: ', chal)

                ret_data = np.hstack([list_files, dummy_anno, np.arange(len(list_files)).reshape(-1, 1)])
                chal_test_dataset[vid] = ret_data

            np.save('{}/{}_test.npy'.format(dataset_path, chal), chal_test_dataset)
            print('Completed: ', chal)
