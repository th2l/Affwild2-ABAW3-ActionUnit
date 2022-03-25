import os.path

import numpy as np
import pandas as pd

if __name__ == '__main__':
    root_folder = '/mnt/Work/Dataset/Affwild2_ABAW3/submissions/AU_Detection_final'
    with open('./dataset/testset/AU_test_set_release.txt') as fd:
        list_file = fd.read().splitlines()
        write_folder = os.path.join(root_folder, '5')
        os.makedirs(write_folder, exist_ok=True)
        header_name = ['AU1', 'AU2', 'AU4', 'AU6', 'AU7', 'AU10', 'AU12', 'AU15', 'AU23', 'AU24', 'AU25', 'AU26']

        for fname in list_file:
            f1 = pd.read_csv(os.path.join(root_folder, '1', fname+'.txt')).values
            f2 = pd.read_csv(os.path.join(root_folder, '2', fname + '.txt')).values
            f3 = pd.read_csv(os.path.join(root_folder, '3', fname + '.txt')).values
            f4 = pd.read_csv(os.path.join(root_folder, '4', fname + '.txt')).values
            f_fused = np.stack([f1, f2, f3, f4])
            f_fused_sum = np.sum(f_fused, axis=0)
            f_fused_ = (f_fused_sum >= 2).astype(int)
            print(f_fused.shape, f_fused_sum.shape, f_fused_.shape)

            pd.DataFrame(data=f_fused_, columns=header_name).to_csv(
                '{}/{}.txt'.format(write_folder, fname), index=False)

    print('Finished fusion')


