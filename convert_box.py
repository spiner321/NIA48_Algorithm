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
import traceback
from tqdm.notebook import tqdm as tqdm_nb
from tqdm import tqdm

import multiprocessing as mp
from functools import partial


class Fit3dBox:

    def __init__(self, bf_label, bf_pcd, bf_id, tf_label, tf_pcd, weather):
        self.bf_anns = bf_label['annotation']
        self.bf_pcd = bf_pcd
        self.bf_id = bf_id
        self.tf_anns = tf_label['annotation']
        self.tf_pcd = tf_pcd
        self.weather = weather

        # Best Frame의 객체 id와 box index를 매칭
        bf_id_idx = {}
        for i, bf_ann in enumerate(self.bf_anns):
            bf_id_idx[bf_ann['id']] = i
        self.bf_id_idx = bf_id_idx

    
    def subbox_select(self, subbox_list, category):
        if category == 'OVERPASS' or category == 'TUNNEL':
            y_loc = {}
            for i, subbox in enumerate(subbox_list):
                y_loc[i] = subbox['location'][1]
            idx_yloc = sorted(y_loc.items(), key=lambda x: x[1], reverse=True)

            left_box = subbox_list[idx_yloc[0][0]]
            center_box = subbox_list[idx_yloc[1][0]]
            right_box = subbox_list[idx_yloc[2][0]]
            
            return left_box, center_box, right_box


    def find_floor(self, pcd_in_dim, location):
        temp_pcd_in_dim = copy.deepcopy(pcd_in_dim)
        temp_pcd_in_dim = temp_pcd_in_dim[(temp_pcd_in_dim[:, 2] <= location[2])]
        x_cord = np.around(temp_pcd_in_dim[:, 0], 1)
        temp_pcd_in_dim[:, 0] = x_cord

        floor = []
        for x_dot in set(x_cord):
            z_dot = temp_pcd_in_dim[temp_pcd_in_dim[:, 0]==x_dot, 2]
            floor.append(np.min(z_dot))

        while True:
            # 배열의 기울기 계산
            gradients = np.diff(floor)

            if len(gradients) == 0 or abs(np.max(gradients)) <= 0.1:
                break

            # 기울기 값이 크게 변하는 구간 제외
            outliers_idx = np.where(abs(gradients) > 0.1)[0]

            del_idx = []
            for i in outliers_idx:
                if floor[i] >= floor[i+1]:
                    del_idx.append(i)
                else:
                    del_idx.append(i+1)
            floor = np.delete(floor, list(set(del_idx)))

        floor_point = np.max(floor) + 0.05
        
        return floor_point


    def road_sign(self):
        bf_anns = self.bf_anns
        bf_pcd = self.bf_pcd
        bf_id = self.bf_id
        tf_anns = self.tf_anns
        tf_pcd = self.tf_pcd
        weather = self.weather

        bf_id_idx = self.bf_id_idx

        matching_id = bf_id_idx[bf_id]
        bf_box = bf_anns[matching_id]['3d_box'][0]

        ## bf_box 상자 크기 포인트클라우드에 맞추기
        bf_location = bf_box['location']
        bf_dimension = bf_box['dimension'] # w, h, l
        bf_rotation_y = bf_box['rotation_y']

        if 'Rain' not in weather:
            bf_extra_range = [0, 1, 0] # w, h, l에 더할 값
        else:
            bf_extra_range = [0, 0, 0]

        bf_extra_dim = np.array(bf_dimension) + np.array(bf_extra_range)

        bf_rmat, bf_rmat_inv = self._rmat_and_inv(bf_rotation_y)
        
        bf_corners_3d = self._get_3d_corners(bf_location, bf_extra_dim, bf_rmat)
        bf_pcd_in_dim = self._get_pcd_in_dim(bf_pcd, bf_corners_3d, bf_rmat_inv)

        # 박스 범위 조정
        bf_x_min = np.min(bf_pcd_in_dim[:,0])
        bf_x_max = np.max(bf_pcd_in_dim[:,0])
        bf_y_min = np.min(bf_pcd_in_dim[:,1])
        bf_y_max = np.max(bf_pcd_in_dim[:,1])
        bf_z_min = np.min(bf_pcd_in_dim[:,2])
        bf_z_max = np.max(bf_pcd_in_dim[:,2])

        w = abs(bf_y_max - bf_y_min)
        h = abs(bf_z_max - bf_z_min)
        l = abs(bf_x_max - bf_x_min) + 0.1

        x = bf_x_min + l/2
        y = bf_y_min + w/2
        z = bf_z_min + h/2

        bf_box['dimension'] = [w, h, l]
        bf_box['location'] = [x, y, z]
        # bf_box['location'] = np.dot(bf_rmat, np.array([x, y, z]).T).tolist()
        
        for tf_ann in tf_anns:
            if tf_ann['category']=='ROAD_SIGN' and tf_ann['id']==bf_id:
            # if tf_ann['category']=='ROAD_SIGN' and tf_ann['id'] in bf_id_idx.keys():
                # matching_id = bf_id_idx[tf_ann['id']]

                tf_box = tf_ann['3d_box'][0]

                tf_box['dimension'] = bf_box['dimension']

                location = tf_box['location']
                dimension = tf_box['dimension'] # w, h, l
                rotation_y = tf_box['rotation_y']

                extra_range = [0, 1, 0.5] # w, h, l에 더할 값
                extra_dim = np.array(dimension) + np.array(extra_range)

                rmat, rmat_inv = self._rmat_and_inv(rotation_y)
                
                corners_3d = self._get_3d_corners(location, extra_dim, rmat)
                pcd_in_dim = self._get_pcd_in_dim(tf_pcd, corners_3d, rmat_inv)

                # 원점에서 가장 가까운 x값에 근접하도록 박스를 이동
                x_dim_min = location[0] - extra_dim[2]/2
                x_pcd_min = np.min(pcd_in_dim[:,0])
                x_diff = x_dim_min - x_pcd_min
                    
                if x_diff >= 0: # pcd가 dimension 안에 속한 경우
                    tf_box['location'][0] = location[0] - extra_range[2]/2 + x_diff - 0.01 # extra_range 영향 제거 / 선과 pcd가 닿지 않도록 해주는 여유값 추가 (0.01m)
                else: # pcd가 dimension 밖에 있는 경우
                    tf_box['location'][0] = location[0] - extra_range[2]/2 - x_diff - 0.01

                # 원점에서 가장 가까운 z값에 근접하도록 박스를 이동, weather에 따라 다르게 적용
                if 'Rain' not in weather:
                    z_dim_min = location[2] - extra_dim[1]/2
                    z_pcd_min = np.min(pcd_in_dim[:,2])
                    z_diff = z_dim_min - z_pcd_min

                    if z_diff >= 0: # pcd가 dimension 안에 속한 경우
                        tf_box['location'][2] = location[2] - extra_range[1]/2 + z_diff - 0.01
                    else: # pcd가 dimension 밖에 있는 경우
                        tf_box['location'][2] = location[2] - extra_range[1]/2 - z_diff - 0.01

        return tf_anns

    
    def tunnel(self):

        def find_edge(location, pcd_in_dim, rmat_inv, floor_point, pillar_loc: str='left'):
            margin = 0

            while True:
                try:
                    temp_pcd_in_dim = copy.deepcopy(pcd_in_dim)
                    temp_pcd_in_dim = temp_pcd_in_dim[(temp_pcd_in_dim[:, 2] > floor_point) & \
                                                    (temp_pcd_in_dim[:, 0] >= location[0] - 5) & (temp_pcd_in_dim[:, 0] <= location[0] + 5)]
                    # front_box = temp_pcd_in_dim[:, [1, 2]]
                    z_cord = np.around(temp_pcd_in_dim[:, 2], 1)
                    temp_pcd_in_dim[:, 2] = z_cord

                    left_edge = []
                    right_edge = []

                    # 연석 찾기
                    for z_dot in sorted(set(z_cord))[:10]:
                        y_dot = temp_pcd_in_dim[temp_pcd_in_dim[:, 2]==z_dot, 1]
                        if pillar_loc == 'left':
                            right_edge.append(np.min(y_dot))
                        elif pillar_loc == 'right':
                            left_edge.append(np.max(y_dot))
                            
                    # 터널 벽면 찾기
                    for z_dot in sorted(set(z_cord))[10:30]:
                        y_dot = temp_pcd_in_dim[temp_pcd_in_dim[:, 2]==z_dot, 1]
                        if pillar_loc == 'left':
                            left_edge.append(np.min(y_dot))
                        elif pillar_loc == 'right':
                            right_edge.append(np.max(y_dot))
                    
                    # if pillar_loc == 'left':
                    #     left_point = np.max(left_edge)
                    #     right_point = np.min(right_edge)
                    # elif pillar_loc == 'right':
                    #     left_point = np.max(left_edge)
                    #     right_point = np.min(right_edge)
                    left_point = np.max(left_edge)
                    right_point = np.min(right_edge)

                    # 터널 높이 정하기
                    if pillar_loc == 'left':
                        top_edge = pcd_in_dim[(pcd_in_dim[:, 1]>=right_point - margin) & (pcd_in_dim[:, 1]<=right_point + margin) & \
                                            (pcd_in_dim[:, 0]>=location[0] - 5) & (pcd_in_dim[:, 0]<=location[0] + 5) & \
                                            (pcd_in_dim[:, 2]>location[2]), 2]
                        top_point = np.min(top_edge)
                    elif pillar_loc == 'right':
                        top_edge = pcd_in_dim[(pcd_in_dim[:, 1]>=left_point - margin) & (pcd_in_dim[:, 1]<=left_point + margin) & \
                                            (pcd_in_dim[:, 0]>=location[0] - 5) & (pcd_in_dim[:, 0]<=location[0] + 5) & \
                                            (pcd_in_dim[:, 2]>location[2]), 2]
                        top_point = np.min(top_edge)

                    if len(left_edge) > 0 and len(right_edge) > 0 and len(top_edge) > 0:
                        break

                except:
                    margin += 0.05
                    pass
            
            left_point = np.dot(rmat_inv, pcd_in_dim[pcd_in_dim[:, 1]==left_point].T)[1, 0]
            right_point = np.dot(rmat_inv, pcd_in_dim[pcd_in_dim[:, 1]==right_point].T)[1, 0]

            if pillar_loc == 'left':
                left_point += 0.2
            elif pillar_loc == 'right':
                right_point -= 0.2

            return left_point, right_point, top_point
        
        
        bf_anns = self.bf_anns
        bf_pcd = self.bf_pcd
        bf_id = self.bf_id
        tf_anns = self.tf_anns
        tf_pcd = self.tf_pcd
        weather = self.weather

        bf_id_idx = self.bf_id_idx
        matching_id = bf_id_idx[bf_id]

        bf_box = bf_anns[matching_id]['3d_box']

        ### bf_box 상자 크기 포인트클라우드에 맞추기
        left_bf_box, center_bf_box, right_bf_box = self.subbox_select(bf_box, 'TUNNEL')

        # 원본 좌표 저장
        # left_location_ori = left_bf_box['location']
        center_location_ori = center_bf_box['location']
        # right_location_ori = right_bf_box['location']

        ## 터널 양쪽 기둥
        pillar_top_points = []
        for pillar_loc, pillar_bf_box in zip(['left', 'right'], [left_bf_box, right_bf_box]):
            location = pillar_bf_box['location']
            dimension = pillar_bf_box['dimension'] # w, h, l
            rotation_y = pillar_bf_box['rotation_y']

            extra_range = [3, 10, 0] # w, h, l에 더할 값
            extra_dim = np.array(dimension) + np.array(extra_range)

            rmat, rmat_inv = self._rmat_and_inv(rotation_y)

            corners_3d = self._get_3d_corners(location, extra_dim, rmat)
            pcd_in_dim = self._get_pcd_in_dim(bf_pcd, corners_3d, rmat_inv)

            # 지면 찾기
            floor_point = self.find_floor(pcd_in_dim, location)

            # 기둥 범위 정하기
            left_point, right_point, top_point = find_edge(location, pcd_in_dim, rmat_inv, floor_point, pillar_loc=pillar_loc)

            w = abs(left_point - right_point)
            h = abs(top_point - floor_point)

            y = left_point - w/2
            z = top_point - h/2

            pillar_bf_box['location'] = [location[0], y, z]
            pillar_bf_box['dimension'] = [w, h, dimension[2]]

            pillar_top_points.append(top_point)

        # 기둥 높이 동일하게 맞추기
        if pillar_top_points[0] >= pillar_top_points[1]: # 왼쪽 기둥이 더 높은 경우
            pillar_top_point = np.max(pillar_top_points)
            pillar_bottom_point = right_bf_box['location'][2] - right_bf_box['dimension'][1]/2

            h = abs(pillar_top_point - pillar_bottom_point)
            right_bf_box['dimension'][1] = h
            right_bf_box['location'][2] = pillar_bottom_point + h/2

        elif pillar_top_points[0] < pillar_top_points[1]: # 오른쪽 기둥이 더 높은 경우
            pillar_top_point = np.max(pillar_top_points)
            pillar_bottom_point = left_bf_box['location'][2] - left_bf_box['dimension'][1]/2

            h = abs(pillar_top_point - pillar_bottom_point)
            left_bf_box['dimension'][1] = h
            left_bf_box['location'][2] = pillar_bottom_point + h/2


        ## 터널 상단
        y_max = left_bf_box['location'][1] + left_bf_box['dimension'][0]/2
        y_min = right_bf_box['location'][1] - right_bf_box['dimension'][0]/2
        z_max = center_bf_box['location'][2] + center_bf_box['dimension'][1]/2
        z_min = pillar_top_point

        w = abs(y_max - y_min)
        h = abs(z_max - z_min)

        x = center_bf_box['location'][0]
        y = y_min + w/2
        z = z_min + h/2

        center_bf_box['location'] = [x, y, z]
        center_bf_box['dimension'] = [w, h, center_bf_box['dimension'][2]]

        bf_anns[matching_id]['3d_box'] = [left_bf_box, center_bf_box, right_bf_box]

        # mov_left_loc = np.asarray(left_bf_box['location']) - np.asarray(left_location_ori)
        mov_center_loc = np.asarray(center_bf_box['location']) - np.asarray(center_location_ori)
        # mov_right_loc = np.asarray(right_bf_box['location']) - np.asarray(right_location_ori)


        ## 타겟 프레임 객체 위치, 크기 조정
        for tf_ann in tf_anns:
            if tf_ann['category']=='TUNNEL' and tf_ann['id']==bf_id:

                tf_box = tf_ann['3d_box']
                left_tf_box, center_tf_box, right_tf_box = self.subbox_select(tf_box, 'TUNNEL')

                left_tf_box['dimension'] = left_bf_box['dimension']
                center_tf_box['dimension'] = center_bf_box['dimension']
                right_tf_box['dimension'] = right_bf_box['dimension']

                # left_tf_box['location'] = (np.asarray(left_tf_box['location']) + mov_left_loc).tolist()
                center_tf_box['location'] = (np.asarray(center_tf_box['location']) + mov_center_loc).tolist()
                # right_tf_box['location'] = (np.asarray(right_tf_box['location']) + mov_right_loc).tolist()

                left_tf_box['location'][1] = center_tf_box['location'][1] + center_tf_box['dimension'][0]/2 - left_tf_box['dimension'][0]/2
                left_tf_box['location'][2] = center_tf_box['location'][2] - center_tf_box['dimension'][1]/2 - left_tf_box['dimension'][1]/2
                right_tf_box['location'][1] = center_tf_box['location'][1] - center_tf_box['dimension'][0]/2 + right_tf_box['dimension'][0]/2
                right_tf_box['location'][2] = center_tf_box['location'][2] - center_tf_box['dimension'][1]/2 - right_tf_box['dimension'][1]/2

                tf_ann['3d_box'] = [left_tf_box, center_tf_box, right_tf_box]
      
        return tf_anns



class PointHandler:

    def _rmat_and_inv(self, rotation_y):
        euler_angle = [0,0, rotation_y]
        rot = R.from_euler('xyz', euler_angle, degrees=True)  # type: ignore
        rot_inv = rot.inv()
        rmat = np.array(rot.as_matrix())
        rmat_inv = np.array(rot_inv.as_matrix())

        return rmat, rmat_inv
    

    def _get_3d_corners(self, location, dimension, rmat):
        '''location: [x, y, z]
           dimension: [w, h, l]
           rmat: rotation matrix
           extra_range: [l, w, h]에 추가로 더할 값'''

        x = location[0]
        y = location[1]
        z = location[2]
        l = dimension[2]
        w = dimension[0]
        h = dimension[1]

        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        z_corners = [h/2, h/2, h/2, h/2, -h/2, -h/2, -h/2, -h/2]
        corners_3d = np.dot(rmat, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d[0, :] = corners_3d[0, :] + x
        corners_3d[1, :] = corners_3d[1, :] + y
        corners_3d[2, :] = corners_3d[2, :] + z

        return corners_3d


    def _get_pcd_in_dim(self, tf_pcd, corners_3d, rmat_inv):
        x_min = np.min(corners_3d[0, :])
        x_max = np.max(corners_3d[0, :])
        y_min = np.min(corners_3d[1, :])
        y_max = np.max(corners_3d[1, :])
        z_min = np.min(corners_3d[2, :])
        z_max = np.max(corners_3d[2, :])

        pcd_range = tf_pcd[(tf_pcd[:,0] >= x_min) & (tf_pcd[:,0] <= x_max) & \
                            (tf_pcd[:,1] >= y_min) & (tf_pcd[:,1] <= y_max) & \
                            (tf_pcd[:,2] >= z_min) & (tf_pcd[:,2] <= z_max)]

        pcd_in_dim = np.dot(rmat_inv, pcd_range.T)

        return pcd_in_dim.T



class ConvertBox(PointHandler, Fit3dBox):

    def __init__(self, scene_path, bf_num, bf_id, tf_num, categories):
        '''scene_path: pcd가 저장된 경로
            bf_num: best frame 번호
            tf_num: target frame 번호
            categories: 변환할 카테고리 리스트'''

        self.scene_path = scene_path
        self.lidar_path = os.path.join(self.scene_path, 'Lidar/*.pcd')
        # self.calib_path = os.path.join(self.scene_path, 'calib/Lidar_camera_calib/*.txt')
        # self.label_path = os.path.join(self.scene_path, 'result/*.json').replace('source', 'label')
        self.label_path = os.path.join(self.scene_path, 'result/*.json').replace('raw', 'new_label').replace('source', 'label')
        self.meta_path = os.path.join(self.scene_path, 'Meta/*.json')
        
        self.bf_num = bf_num - 1
        self.bf_id = bf_id
        self.tf_num = tf_num - 1
        
        self.categories = categories

        self.bf_pcd, self.tf_pcd = self.get_pcd()
        self.bf_label, self.tf_label, self.weather = self.get_label()
        
        super().__init__(self.bf_label, self.bf_pcd, self.bf_id, self.tf_label, self.tf_pcd, self.weather)


    def get_pcd(self):
        bf_pcd_path = sorted(glob.glob(self.lidar_path))[self.bf_num]
        tf_pcd_path = sorted(glob.glob(self.lidar_path))[self.tf_num]
        
        bf_pcd = o3d.io.read_point_cloud(bf_pcd_path)
        bf_pcd = np.asarray(bf_pcd.points)
        bf_pcd = np.delete(bf_pcd, np.where((bf_pcd[:,0] < 0) | (bf_pcd[:,0] > 80)), 0)
    
        tf_pcd = o3d.io.read_point_cloud(tf_pcd_path)
        tf_pcd = np.asarray(tf_pcd.points)
        tf_pcd = np.delete(tf_pcd, np.where((tf_pcd[:,0] < 0) | (tf_pcd[:,0] > 80)), 0)

        return bf_pcd, tf_pcd


    def get_label(self):
        bf_label_path = sorted(glob.glob(self.label_path))[self.bf_num]
        tf_label_path = sorted(glob.glob(self.label_path))[self.tf_num]
        meta_path = sorted(glob.glob(self.meta_path))[0]

        with open(bf_label_path, 'r') as f:
            bf_label = json.load(f)
        with open(tf_label_path, 'r') as f:
            tf_label = json.load(f)
        with open(meta_path, 'r') as f:
            meta = json.load(f)

        weather = meta['weather']

        # bf_anns = bf_label['annotation']
        # tf_anns = tf_label['annotation']

        return bf_label, tf_label, weather


    def converting(self):

        if 'ROAD_SIGN' in self.categories:
            new_tf_anns = self.road_sign()
            self.tf_label['annotation'] = new_tf_anns

        elif 'TUNNEL' in self.categories:
            new_tf_anns = self.tunnel()
            self.tf_label['annotation'] = new_tf_anns

        self.save_new_label()


    def save_new_label(self):
        # # 수정한 라벨 저장 위치
        # new_label_path = os.path.join(self.scene_path, 'result/').replace('source', 'new_label')
        # os.makedirs(os.path.dirname(new_label_path), exist_ok=True)

        # # 베스트 프레임 라벨 복사
        # bf_label_path = sorted(glob.glob(self.label_path))[self.bf_num]
        # new_bf_label_path = bf_label_path.replace('label', 'new_label')
        # shutil.copy(bf_label_path, new_bf_label_path)

        # # 수정한 타겟 프레임 라벨 저장
        # tf_label_path = sorted(glob.glob(self.label_path))[self.tf_num]
        # new_tf_label_path = tf_label_path.replace('label', 'new_label')
        # with open(new_tf_label_path, 'w') as f:
        #     json.dump(self.tf_label, f, indent=4)

        # 수정한 타겟 프레임 라벨 저장 (원본 라벨 덮어쓰기)
        tf_label_path = sorted(glob.glob(self.label_path))[self.tf_num]
        with open(tf_label_path, 'w') as f:
            json.dump(self.tf_label, f, indent=4)


if __name__ == '__main__':
    bf_df = pd.read_csv('/data/kimgh/NIA48_Algorithm/bestframe_roadsign_tunnel.csv', index_col=0)
    bf_df = bf_df.sort_values(by=['category', 'clipname']).reset_index(drop=True)
    bf_df = bf_df[['clipname', 'bestframe', 'id', 'category']]
    bf_df = bf_df.loc[bf_df['category']=='TUNNEL'].reset_index(drop=True)

    start = time.time()

    os.makedirs('errors', exist_ok=True)

    def convert_box(scene, bf_num, bf_id, categories):
        # print(f'Converting Start - [scene: {scene} / bf: {bf_num} / bf_id: {bf_id} / category: {categories}]')
        global bf_df
        global start

        scene_path = glob.glob(f'/data/NIA48/raw/*/source/*/*/{scene}')[0]

        frames = sorted(os.listdir(os.path.join(scene_path, 'Lidar')))
        frames = np.arange(1, len(frames)+1, 1)
        frames = np.hstack([frames[frames!=bf_num], bf_num])
        
        # for tf_num in tqdm(frames, desc=f'[scene: {scene} / bf: {bf_num} / bf_id: {bf_id} / category: {categories}]'):
        for tf_num in frames:
            try:
                ConvertBox(scene_path, bf_num, bf_id, tf_num, categories).converting()
            except:
                error = f'{scene},{bf_num},{tf_num},{bf_id},{categories}\n'
                with open('errors/error_ls.txt', 'a') as f:
                    f.write(error)

                with open(f'errors/{scene}.txt', 'a') as f:
                    f.write('--------------------------------------------------------------------------------\n')
                    f.write(f'Error - [Scene: {scene}  Bf: {bf_num} / Tf: {tf_num} / Bf_id: {bf_id} / Category: {categories}]\n')
                    f.write(traceback.format_exc())
                    f.write('--------------------------------------------------------------------------------\n')

                print(f'Error - [Scene: {scene}  Bf: {bf_num} / Tf: {tf_num} / Bf_id: {bf_id} / Category: {categories}]')

        delta_time = time.strftime('%H:%M:%S', time.gmtime(time.time()-start))

        done_idx = bf_df.loc[(bf_df['clipname']==scene)].index[0] + 1
        print(f'Converting Done - [{done_idx}/{len(bf_df)}][Tot: {delta_time}] [Scene: {scene} | Bf: {bf_num} | Bf_id: {bf_id} | Category: {categories}]')
        
    # scenes = bf_df.values[:, 0]
    # bf_nums = bf_df.values[:, 1]
    # bf_ids = bf_df.values[:, 2]
    # categories = bf_df.values[:, 3]
    # inputs = zip(scenes, bf_nums, bf_ids, categories)

    with mp.Pool(processes=40) as pool:
        # pool.starmap(convert_box, inputs)
        pool.starmap(convert_box, bf_df.values)
        pool.close()
        pool.join()