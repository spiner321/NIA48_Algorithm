if __name__ == '__main__':
    # from proj_2d_box import PointHandler
    import json
    from scipy.spatial.transform import Rotation as R
    import numpy as np
    import pandas as pd
    import os
    import shutil
    import glob
    import re
    import open3d as o3d
    import math
    import copy
    import time
    import pickle
    from tqdm.notebook import tqdm as tqdm_nb
    from tqdm.auto import tqdm as tqdm_auto
    from tqdm import tqdm

    import multiprocessing as mp
    from functools import partial

    # 원본 라벨 복사

    # labels = glob.glob('/data/NIA48/raw/*/label/*/*/*/result/*.json')
    with open('/data/kimgh/NIA48_Algorithm/labels_origin.pkl', 'rb') as f:
        labels = pickle.load(f)

    def copy_label(label):
        # for label in tqdm(labels):
        # print(label)
        new_label = label.replace('raw', 'new_label')
        os.makedirs(os.path.dirname(new_label), exist_ok=True)
        shutil.copy(label, new_label)

    with mp.Pool(processes=48) as pool:
        pool.map(copy_label, tqdm(sorted(labels)))
        pool.close()
        pool.join()