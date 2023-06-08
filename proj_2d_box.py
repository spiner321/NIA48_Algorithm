import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import shapely
import open3d as o3d



FIT_LIM_DICT = {            
    "CAR":          [[1.5, 1.5, 3.5], [2.5, 2.5, 4.0]],
    "MOTORCYCLE":   [[0.8, 1.5, 1.5], [1.2, 1.5, 2.0]],
    "BUS":          [[2.0, 2.8, 7.0], [2.5, 3.5, 12.5]],
    "BICYCLE":      [[0.8, 1.5, 1.5], [1.2, 1.5, 2.0]],
    "TRUCK":        [[1.5, 2.0, 5.0], [2.5, 3.5, 12.0]],
    "PEDESTRIAN":   [[0.5, 1.5, 0.3], [0.8, 2.0, 1.0]],
    "ETC":          [[-1, -1, -1], [-1, -1, -1]],

    'MEDIAN_STRIP': [[-1, -1, -1],[-1, -1, -1]],
    'OVERPASS': [[-1, -1, -1],[-1, -1, -1]],
    'TUNNEL': [[-1, -1, -1],[-1, -1, -1]],
    'SOUND_BARRIER': [[-1, -1, -1],[-1, -1, -1]],
    'STREET_TREES': [[-1, -1, -1],[-1, -1, -1]],
    'RAMP_SECT': [[-1, -1, -1],[-1, -1, -1]],
    'ROAD_SIGN': [[-1, -1, -1],[-1, -1, -1]]
}

PT_IDX_DICT = {
    "1": [1,1,8,7,2,5],
    "2": [2,2,7,8,1,6],
    "3": [3,3,6,5,4,7],
    "4": [4,4,5,6,3,8]
} ##//val: [ x_t, y_t, x_b, y_b, intercept_uppper, intercept_lower ]

CAM_FOV_DICT = {
    "S_former": [0.97307786,0.002598030572050547, -0.98814101,-0.001989300997648158],
    "S_recent": [0.90035597, 0.026340484628278205, -1.06257365, -0.012752284734313335],
    "A": [0.96583887, -0.19539506935009143, -0.99438957, 0.15072875651308948]
    
} ##//val: [ coeff_ypos, intercept_ypos, coeff_yneg, intercept_yneg ]

# apply camera changes
S_CLIP_PIVOT = 28493

FIX_DIM = {
    "CAR": [-1, -1, -1],
    "MOTORCYCLE": [-1, -1, -1],
    "BUS": [-1, -1, -1],
    "BICYCLE": [-1, -1, -1],
    "TRUCK": [-1, -1, -1],
    "PEDESTRIAN": [-1, -1, -1],
    "ETC": [-1, -1, -1],

    'MEDIAN_STRIP': [-1, -1, -1],
    'OVERPASS': [-1, -1, -1],
    'TUNNEL': [-1, -1, -1], 
    'SOUND_BARRIER': [-1, -1, -1],
    'STREET_TREES': [-1, -1, -1],
    'RAMP_SECT': [-1, -1, -1], 
    'ROAD_SIGN': [-1, -1, -1]
}

class PointHandler:
    def __init__(self, lidarpath):
        self.lidar_pts = self.parse_pcd(lidarpath)
        # self.inbox_pcd = None

    def parse_pcd(self, lidarpath):
        #// parse *.pcd file to np.array
        pcd_load = o3d.io.read_point_cloud(lidarpath)
        ret = np.asarray(pcd_load.points)
        ret = ret.squeeze()
        
        #// limit lidar pt for performance
        # ret = np.delete(ret, np.where((ret[:,0]<0) | (ret[:,0] > 80)), 0)
        ret = np.delete(ret, np.where((ret[:,0]<1) | (ret[:,0] > 80)), 0)
        # ret = np.delete(ret, np.where((ret[:,1]<-40) | (ret[:,1] > 40)), 0)
        
        return ret

    
    def minmax_pcd(self, pcd_arr):
        # if self.inbox_pcd is None: return None
        #// min max values of x, y, z
        pcd_np = np.array(pcd_arr)
        
        min_vals = np.min(pcd_np, axis=0)
        max_vals = np.max(pcd_np, axis=0)
        return min_vals, max_vals

    def is_presumed(self, before, after, pivot=0.5):
        if after<before*pivot:
            return True
        else:
            return False








    def act_(self, center, size, rotation, fit_target="whl", category="",bf_loc=[], bf_dim=[], action=None):
        """
        action notation
        - pass              0
        - bestframe-fit     1
        - targetframe-fit   2
        - dim-variable      3
        """
        
        if action==0:
            return center, size
        if action==1:
            fit_center, fit_size = self.action_1(center, size, rotation, 'l', category,bf_loc, bf_dim)
        if action==2:
            fit_center, fit_size = self.action_2(center, size, rotation, "whl", category, bf_loc, bf_dim, offset_z=0.5)
            pass
        return


    def action_1(self, center, size, rotation, fit_target, category, bf_loc, bf_dim):
        label_center = center.copy()
        label_size = size.copy()

        label_length, bf_length = label_size[2], bf_dim[2]
        if abs(label_length - bf_length) > 0.3: 
            return bf_loc, bf_dim
        else:
            fit_center, fit_size, flag = self._fit_at_origin(bf_loc, bf_dim, rotation, fit_target='l', category=category)
            if flag==False or flag==None:
                return bf_loc, bf_dim

            dim_limit = FIT_LIM_DICT.get(category)
            min_limit, max_limit = dim_limit[0], dim_limit[1] # type: ignore
            length = fit_size[2]
            min_length = min_limit[2]
            if length < min_length:
                delta_x = abs(length-min_length)/2
                fit_size[2] = min_length
                fit_center[0] = fit_center[0] + delta_x
                # bf_loc[0] -= delta_x/2
            fit_center, fit_size = self._restore(bf_loc, fit_center, fit_size, rotation)
            return fit_center, fit_size




    def action_2(self, center, size, rotation, fit_target, category, bf_loc, bf_dim, offset_z=0.1):
        label_center = center.copy()
        label_size = size.copy()
        
        presumed_loc_z = self.presume_floor(bf_loc, bf_dim, center, size)

        label_center[2] = presumed_loc_z
        label_center[2] += offset_z
        
        fit_center_orig, fit_size, flag = self._fit_at_origin(label_center, label_size, rotation, fit_target, category)
        
        if flag==False or flag==None:
            return label_center, bf_dim
        shifted_loc = self.shift_loc(label_center, fit_center_orig, fit_size, bf_dim, rotation)
        
        #// RESTORE DELTA
        # loc_delta = [b-a for b,a in zip(fit_center_orig,shifted_loc)]
        loc_delta = [a-b for b,a in zip(fit_center_orig,shifted_loc)]

        label_center = [lb+d for lb,d in zip(label_center,loc_delta)]
        
        fit_center, fit_size = self._restore(label_center, fit_center_orig, fit_size, rotation)
        

        pcd_num = flag
        step = 0.2
        if center[1] > 0:
            stepnum_x = self.loop_step(fit_center, bf_dim, rotation, category, pcd_num, "x--", step)
            stepnum_y = self.loop_step(fit_center, bf_dim, rotation, category, pcd_num, "y--", step)
            
            if stepnum_x > 1:
                fit_center[0] -= (stepnum_x*step)
                
            if stepnum_y > 1:
                fit_center[1] -= (stepnum_y*step)
                
        else:
            stepnum_x = self.loop_step(fit_center, bf_dim, rotation, category, pcd_num, "x--", step)
            stepnum_y = self.loop_step(fit_center, bf_dim, rotation, category, pcd_num, "y++", step)
            fit_center[0] -= (stepnum_x*step)
            fit_center[1] += (stepnum_y*step)
        
        
        fit_center[2] = presumed_loc_z
        
        return fit_center, bf_dim


    def action_3(self, center, size, rotation, fit_target, category, bf_loc, bf_dim):
        label_center = center.copy()
        label_size = size.copy()

        label_topline = label_center[2] + label_size[1]/2
        # label_botline = label_center[2] - label_size[1]/2
        presumed_loc_z = self.presume_floor(bf_loc, bf_dim, center, size)
        presumed_botline = presumed_loc_z - label_size[1]/2

        label_center[2] = (label_topline + presumed_botline)/2
        label_size[1] = abs(label_topline - presumed_botline)

        fit_center_orig, fit_size, flag = self._fit_at_origin(label_center, label_size, rotation, fit_target, category)
        if flag==False or flag==None:
            return label_center, label_size
        
        fit_center, fit_size = self._restore(label_center, fit_center_orig, fit_size, rotation)
        return fit_center, fit_size






# ======== TARGET FRAME LOGIC ========

    def presume_floor(self, bf_loc, bf_dim, tf_loc, tf_dim, bottom_offset=0.1, z_space=1):
        
        pcd_np = np.array(self.lidar_pts)
        tf_x, tf_y, tf_z = tf_loc.copy()
        tf_w, tf_h, tf_l = tf_dim.copy()
        bf_z, bf_h = bf_loc[2], bf_dim[1]
        box_bottom, box_top = bf_z - bf_h/2, bf_z + bf_h/2
        
        x_range = (pcd_np[:, 0] > tf_x + tf_l / 2) | (pcd_np[:, 0] < tf_x - tf_l / 2)
        y_range = (pcd_np[:, 1] > tf_y + tf_w / 2) | (pcd_np[:, 1] < tf_y - tf_w / 2)
        z_range = (pcd_np[:, 2] > tf_z + tf_h / 2 + z_space) | (pcd_np[:, 2] < tf_z - tf_h / 2 - z_space)
        
        pcd_filtered = np.delete(pcd_np, np.where(x_range | y_range | z_range), axis=0)\
        
        try:
            pcd_floor, pcd_ceiling = np.min(pcd_filtered[:, 2]), np.max(pcd_filtered[:, 2])
        except ValueError:
            pcd_floor, pcd_ceiling = bf_z - bf_h / 2, bf_z + bf_h / 2
        
        if pcd_floor <= box_bottom:
            return pcd_floor + bf_h / 2 + bottom_offset
        else:
            if pcd_ceiling >= box_top:
                return pcd_ceiling - bf_h / 2
            else:
                return bf_z


    def shift_loc(self, origin_center, fit_center, fit_size, target_dim, rotation):
        shifted_loc = [0,0,0]
        
        ref_coord_x = fit_center[0] - (fit_size[2]/2)
        # offset_x = max(fit_size[2]/2, target_dim[2]/2)
        # shifted_loc[0] = ref_coord_x + offset_x
        
        shifted_loc[0] = ref_coord_x + target_dim[2]/2
        
        # ref_coord_z = fit_center[2] + (fit_size[1]/2)
        # shifted_loc[2] = ref_coord_z - target_dim[1]/2

        if origin_center[1] > 0:
            ref_coord_y = fit_center[1] - (fit_size[0]/2)
            shifted_loc[1] = ref_coord_y + target_dim[0]/2
        else:
            ref_coord_y = fit_center[1] + (fit_size[0]/2)            
            shifted_loc[1] = ref_coord_y - target_dim[0]/2
        
        return shifted_loc

    def loop_step(self, fit_center, fit_size, rotation, category, pcd_num, direction, step):
        dim_limit = FIT_LIM_DICT.get(category)
        min_limit, max_limit = dim_limit[0], dim_limit[1] # type: ignore

        step_enum = 0
        prev_loc, prev_dim = fit_center.copy(), fit_size.copy()
        prev_pcd_num = pcd_num
        breakpt = -1 
        # print(pcd_num)
        while(True):
            stepped_center, stepped_size = self.step_cuboid(prev_loc, prev_dim, rotation, direction, step)

            # if direction == "x--":
            #     max_length = max_limit[2]*2
            #                   fit_size[2]*2
            #     curr_length = stepped_size[2]
            #     if curr_length > max_length: 
            #         print(f"breakpoint: max dim {direction}")
            #         break
            # elif direction == "y--":
            #     max_width = max_limit[0]*2     
            #     curr_width = stepped_size[0]      
            #     if curr_width > max_width:
            #         print(f"breakpoint: max dim {direction}") 
            #         break     
            # elif direction == "y++":
            #     max_width = max_limit[0]*2
            #     curr_width = stepped_size[0]
            #     if curr_width > max_width: 
            #         print(f"breakpoint: max dim {direction}")
            #         break



            fit_center_tmp, fit_size_tmp, flag = self._fit_at_origin(stepped_center, stepped_size, rotation)
            if flag == None or flag==False:
                
                print("breakpoint: invalid cuboid")
                break

            curr_pcd_num = flag
            if curr_pcd_num==prev_pcd_num:
                print("breakpoint: pcd num")
                break

            # curr_loc, curr_dim = self._restore(stepped_center, fit_center_tmp, fit_size_tmp, rotation)
            curr_loc, curr_dim = stepped_center, stepped_size

            print(f"step#{step_enum} prev: {prev_loc} {prev_dim}")
            print(f"step#{step_enum} curr: {curr_loc} {curr_dim}")

            prev_loc, prev_dim, prev_pcd_num = curr_loc, curr_dim, curr_pcd_num 

            step_enum += 1

        return step_enum


    def step_cuboid(self, prev_loc, prev_dim, rotation, direction, step):
        stepped_center = prev_loc.copy()
        stepped_size = prev_dim.copy()
        if direction == "x--":
            # min_x = shifted_loc[0] - target_dim[2]/2
            stepped_center[0] -= step/2
            stepped_size[2] += step
        elif direction == "y--":
            # min_y = shifted_loc[1] - target_dim[0]/2
            stepped_center[1] -= step/2
            stepped_size[0] += step
        elif direction == "y++":
            # max_y = shifted_loc[1] + target_dim[0]/2
            stepped_center[1] += step/2
            stepped_size[0] += step
        return stepped_center, stepped_size


    


# ======== FITTING CORE LOGIC ========
    def _fit_at_origin(self, center, size, rotation, fit_target="whl", category=""):
        #// gen Cuboid instance
        cb = Cuboid(center, size, rotation)
        inbox_pcd = cb.inbox_pts(self.lidar_pts.tolist())

        if len(inbox_pcd)<1: 
            return center, size, None

        #// euler to rotation matrix and its inverse 
        rmat, rmat_inv = self._rmat_and_inv(rotation)

        #// rotated & translated pcd
        rot_pcd = []
        for pt in inbox_pcd:
            x_pt,y_pt,z_pt = pt     #// target pt
            x3d, y3d, z3d = center  #// cuboid center

            #// translate to origin
            tmp_x, tmp_y, tmp_z = [ x_pt-x3d, y_pt-y3d, z_pt-z3d ]
            
            #// reshape
            tmp_pt = np.array([[tmp_x, tmp_y, tmp_z]]).reshape((3,1))
            
            #// inverse rotation  
            pt_r = (rmat_inv @ tmp_pt).reshape((3,))
            
            rot_pcd.append(pt_r)

        #// get min max values for dimension
        min_vals, max_vals = self.minmax_pcd(rot_pcd)

        fit_rotation = rotation
        fit_center = [0.0, 0.0, 0.0]    # set origin 
        fit_size = size.copy()                 # default dim
        origin_center = center.copy()
        origin_size = size.copy()              # default dim

        # min max vals of inbox pcd
        xmin, ymin, zmin = min_vals     
        xmax, ymax, zmax = max_vals



        if 'w' in fit_target or 'y' in fit_target: #// fit y-axis (width)
            fit_size[0] = abs(ymin-ymax)
            fit_center[1] = (ymin+ymax)/2

        if 'h' in fit_target or 'z' in fit_target: #// fit z-axis (height)
            fit_size[1] = abs(zmin-zmax)
            fit_center[2] = (zmin+zmax)/2
            
        if 'l' in fit_target or 'x' in fit_target: #// fit x-axis (length)
            fit_size[2] = abs(xmin-xmax)
            fit_center[0] = (xmin+xmax)/2


        #// handle exception 
        if any([k<0.01 for k in fit_size]): return fit_center, fit_size, False
        return fit_center, fit_size, len(inbox_pcd)


    def _restore(self, center, fit_center, fit_size, rotation):
        rmat,_ = self._rmat_and_inv(rotation)
        #// ==== restoring ====

        #// reshape
        tmp_pt = np.array([ fit_center ]).reshape((3,1))
        
        #// rotate at origin
        pt_r = (rmat @ tmp_pt).reshape((3,))

        #// restore from temp translation
        pt_r = pt_r + np.array(center).reshape((3,))

        fit_center = pt_r.tolist()
        return fit_center, fit_size

    def _rmat_and_inv(self, rotation):
        cuboid = Cuboid([0,0,0], [0,0,0], rotation)
        euler_angle = [0,0, cuboid.get_rotation()]
        rot = R.from_euler('xyz', euler_angle, degrees=True)  # type: ignore
        rot_inv = rot.inv()
        rmat = np.array(rot.as_matrix())
        rmat_inv = np.array(rot_inv.as_matrix())

        return rmat, rmat_inv





# ======== CUBOID LOGIC ========

#// 3d cuboid class
class Cuboid:
    def __init__(self, center, size, rotation):
        self.center = center
        self.size = size       
        # self.rotation = rotation*180/math.pi #// rad to deg 변환 
        self.rotation = self.lim_rot(rotation*180/math.pi) #// rad to deg 변환

    def get_center(self):
        return self.center
    def get_size(self):
        return self.size
    def get_rotation(self):
        return self.rotation    
    def set_rotation(self, rotation):
        self.rotation = rotation 

    def lim_rot(self, rot_deg):
        #// limit rotation scope to -90 ~ 90
        if abs(rot_deg) <= 90:
            return rot_deg
        else:
            if rot_deg<0:
                return rot_deg+180
            else:
                return rot_deg-180


    def rotate_pt(self, pt):
        #// rot angle 
        euler_angle = [0,0, self.get_rotation()]
        
        x_pt,y_pt,z_pt = pt
        x3d, y3d, z3d = self.get_center()
        
        #// 원점으로 평행이동 
        tmp_x, tmp_y, tmp_z = [ x_pt-x3d, y_pt-y3d, z_pt-z3d ]
        
        #// euler to rot mat 변환
        temp = R.from_euler('xyz', euler_angle, degrees=True)  # type: ignore
        rmat = np.array(temp.as_matrix())
        tmp_pt = np.array([[tmp_x, tmp_y, tmp_z]]).reshape((3,1))
        
        #// 행렬곱: 회전된 pt 좌표 
        ret = (rmat @ tmp_pt).reshape((3,))
        
        #// 평행이동 복원
        ret = ret + np.array(self.get_center()).reshape((3,))
        return ret



    def get_cuboid_pts(self):
        center = self.get_center()
        size = self.get_size()
        x3d, y3d, z3d = center
    
        w, h, l = size ##//큐보이드 전체 // 반으로 나눠야 중심점으로부터의 거리가 됨
        """
        rot == 0
          3----4
         /|   /|
        1----2 |
        | 7--|-8
        |/   |/
        5----6

        rot == 180
          2----1
         /|   /|
        4----3 |
        | 6--|-5
        |/   |/
        8----7
        """
        #// cuboid 꼭지점 8개 회전
        p1 = [x3d-(l/2), y3d+(w/2), z3d+(h/2)]
        p2 = [x3d-(l/2), y3d-(w/2), z3d+(h/2)]
        p3 = [x3d+(l/2), y3d+(w/2), z3d+(h/2)]
        p4 = [x3d+(l/2), y3d-(w/2), z3d+(h/2)]
        
        p5 = [x3d-(l/2), y3d+(w/2), z3d-(h/2)]
        p6 = [x3d-(l/2), y3d-(w/2), z3d-(h/2)]
        p7 = [x3d+(l/2), y3d+(w/2), z3d-(h/2)]
        p8 = [x3d+(l/2), y3d-(w/2), z3d-(h/2)]
        
        p1 = self.rotate_pt(p1)
        p2 = self.rotate_pt(p2)
        p3 = self.rotate_pt(p3)
        p4 = self.rotate_pt(p4)
        
        p5 = self.rotate_pt(p5)
        p6 = self.rotate_pt(p6)
        p7 = self.rotate_pt(p7)
        p8 = self.rotate_pt(p8)
        
        return p1, p2, p3, p4, p5, p6, p7, p8
        
    def get_cuboid_pts_close(self):
        center = self.get_center()
        size = self.get_size()
        x3d, y3d, z3d = center
        ## l, w, h
        w, h, l = size ##//큐보이드 전체 // 반으로 나눠야 중심점으로부터의 거리가 됨
        
        #// cuboid 꼭지점 내측 8개 회전
        p1 = [x3d-(l/2)+1, y3d+(w/2), z3d+(h/2)]
        p2 = [x3d-(l/2)+1, y3d-(w/2), z3d+(h/2)]
        p3 = [x3d+(l/2)-1, y3d+(w/2), z3d+(h/2)]
        p4 = [x3d+(l/2)-1, y3d-(w/2), z3d+(h/2)]
        
        p5 = [x3d-(l/2)+1, y3d+(w/2), z3d-(h/2)]
        p6 = [x3d-(l/2)+1, y3d-(w/2), z3d-(h/2)]
        p7 = [x3d+(l/2)-1, y3d+(w/2), z3d-(h/2)]
        p8 = [x3d+(l/2)-1, y3d-(w/2), z3d-(h/2)]
        
        p1 = self.rotate_pt(p1)
        p2 = self.rotate_pt(p2)
        p3 = self.rotate_pt(p3)
        p4 = self.rotate_pt(p4)
        
        p5 = self.rotate_pt(p5)
        p6 = self.rotate_pt(p6)
        p7 = self.rotate_pt(p7)
        p8 = self.rotate_pt(p8)
        
        return p1, p2, p3, p4, p5, p6, p7, p8
            
    def inbox_pts(self, lidar_pts: list):
        #// cuboid vertices
        p1, p2, p3, p4, p5, p6, p7, p8 = self.get_cuboid_pts()
        
        #// del z-dimension
        # cb_pts_arr =np.array( [p1, p3, p4, p2] )[:,:2]
        cb_pts_arr =np.array( [p1, p3, p4, p2] )
        min_vals = np.min(cb_pts_arr, axis=0)
        max_vals = np.max(cb_pts_arr, axis=0)        
        xmin, ymin, zmin = min_vals
        xmax, ymax, zmax = max_vals
        

        #// cuboid at top view polygon
        cb_pts = shapely.Polygon(cb_pts_arr)
        
        #// z val minmax
        zmin,zmax = p5[2], p1[2]
        

        #// limit lidar pts for performance
        lidar_np = np.array(lidar_pts)
        lidar_np = np.delete(lidar_np, np.where((lidar_np[:,0]<=xmin) | (lidar_np[:,0] >= xmax)), 0)
        lidar_np = np.delete(lidar_np, np.where((lidar_np[:,1]<=ymin) | (lidar_np[:,1] >= ymax)), 0)
        lidar_np = np.delete(lidar_np, np.where((lidar_np[:,2]<=zmin) | (lidar_np[:,2] >= zmax)), 0)
        
        #// iter lidar points
        ret = []
        for lidar_pt in lidar_np:
            x,y,z = lidar_pt
            pt = shapely.Point([x,y])

            #// if pt is inside cuboid
            if (cb_pts.contains(pt) or cb_pts.touches(pt)) :
                ret.append(lidar_pt)
        
        return ret

    

#// 3d -> 2d projection wrapper 클래스
class Proj_2d_Wrapper:
    def __init__(self, clipname, cuboid):
        self.clipname = clipname
        self.cuboid = cuboid
        self.s_calib_former = {
            "fx": 1044.406012,
            "fy": 1045.640421,
            "cx": 977.767781,
            "cy": 603.580310,

            "k1": -0.120864,
            "k2": 0.057409,
            "p1/k3": 0.000536,
            "p2/k4": -0.000143,

            "rvec": [3.154468,0.041137,0.000315],
            
            "translation_vector": [0.075625,0.000664,-0.224375]
            # "translation_vector": [0.075625,5.000664,-0.224375]
        }
        self.s_calib_recent = {
            "fx": 1059.514412,
            "fy": 1067.573213,
            "cx": 971.806871,
            "cy": 594.326128,

            "k1": -0.143152,
            "k2": 0.063014,
            "p1/k3": -0.000618,
            "p2/k4": 0.000417,

            "rvec": [3.189968,0.035137,0.000565],
            
            "translation_vector": [0.005625,0.000664,-0.204375]
        }
        
        self.a_calib_dict = {
            "fx": 1055.996538,
            "fy": 1053.221700,
            "cx": 950.922296,
            "cy": 587.765542,

            "k1": -0.142402,
            "k2": 0.059090,
            "p1/k3": 0.000053,
            "p2/k4": 0.000071,

            "rvec": [-21.990831, -0.001500, -0.003801],
            "translation_vector": [0.245000,0.037000,-0.177801]
        }
        self.proj_module = self.select_module()

    def select_module(self):
        if self.clipname[0]=="S":
            clipnum_int = int(self.clipname.split('_')[2])
            if clipnum_int<=S_CLIP_PIVOT:
                return Proj2dBox_S_d(self.cuboid, self.s_calib_former, False, CAM_FOV_DICT.get("S_former")) 
            else:
                return Proj2dBox_S_d(self.cuboid, self.s_calib_recent, True, CAM_FOV_DICT.get("S_recent"))
        elif self.clipname[0]=="A":
            return Proj2dBox_A_d(self.cuboid, self.a_calib_dict, CAM_FOV_DICT.get("A"))
        return None


class Proj2dBox_S_d:
    def __init__(self, cuboid, cal_dict, is_recent, cam_fov_val):
        self.cuboid = cuboid
        
        self.calib = cal_dict
        self.is_recent = is_recent
        self.K, self.D, self.euler, self.tvec = self.parse_cal(self.calib)
        self.cam_fov_val = cam_fov_val
        
        self.cuboid_pts = self.mk_cuboid_pts()
        self.cuboid_close_pts = self.mk_cuboid_close_pts()
        self.pts_2d = self.mk_pts_2d()
        
    def set_cuboid(self, cuboid):
        self.cuboid = cuboid
    def get_cuboid(self):
        return self.cuboid

    def get_K(self):
        return self.K
    def get_D(self):
        return self.D
    def get_euler(self):
        return self.euler
    def get_tvec(self):
        return self.tvec
    def get_cuboid_pts(self):
        return self.cuboid_pts
    def get_cuboid_close_pts(self):
        return self.cuboid_close_pts
    def get_pts_2d(self):
        return self.pts_2d

    def mk_cuboid_pts(self):
        cb = self.get_cuboid()
        return cb.get_cuboid_pts()
    def mk_cuboid_close_pts(self):
        cb = self.get_cuboid()
        return cb.get_cuboid_pts_close()
    def mk_pts_2d(self):
        pts_arr = self.get_cuboid_pts()
        pts_close_arr = self.get_cuboid_close_pts()
        
        xi_2d = []
        yi_2d = []
        xi_close_2d = []
        yi_close_2d = []
        
        for pt, pt_close in zip(pts_arr, pts_close_arr):
            x, y, z = pt
            x_close, y_close, z_close = pt_close
            xy_2d = self.zxy_to_xy(z, x, y, self.get_tvec(), self.get_euler(), self.get_K(), self.get_D())
            xy_2d_close = self.zxy_to_xy(z_close, x_close, y_close, self.get_tvec(), self.get_euler(), self.get_K(), self.get_D())
            xi_2d.append(xy_2d[0][0])
            yi_2d.append(xy_2d[0][1])
            xi_close_2d.append(xy_2d_close[0][0])
            yi_close_2d.append(xy_2d_close[0][1])

        return [xi_2d, yi_2d, xi_close_2d, yi_close_2d]    

    def get_border_pt(self, x1, y1, x2, y2):
        a, b = 0, 0
        if x1 != x2:
            a = (y2-y1)/(x2-x1)
            b = y1 - (a*x1)
        return a, b
    
    def limit_val(self, k, maxres, minres=0):
        if k < minres: k = minres
        elif k>maxres: k = maxres
        return k
    
    def proj_2d_normal(self):
        """_summary_
        발산 보정 미적용 2d box 값  
        Returns:
            _list_: _[[x,y,w,h], area]_
        """        
        cb = self.get_cuboid()
        loc, dim, rot = cb.get_center(), cb.get_size(), abs(cb.get_rotation())
        
        box_2d, area_2d = [], 0
        pts_2d = self.get_pts_2d()
        xi_2d = pts_2d[0]
        yi_2d = pts_2d[1]
        xmin, xmax, ymin, ymax = self.limit_val(min(xi_2d), 1920), self.limit_val(max(xi_2d), 1920), \
                                 self.limit_val(min(yi_2d), 1200), self.limit_val(max(yi_2d), 1200)
        center_x, center_y, w, h = (xmax+xmin)/2, (ymax+ymin)/2, abs(xmax-xmin), abs(ymax-ymin)
        box_2d = [center_x, center_y, w, h]
        area_2d = w*h
        
        return [box_2d, area_2d]
    
    def correct_2d_indev(self, semantical=False):
        """_summary_

        Args:
            semantical (bool, optional): _True:근거리 의미적 보정 적용; False:단순 발산 보정만 적용_. Defaults to False.

        Returns:
            _list_: _[[x,y,w,h], area]_
        """
        cb = self.get_cuboid()
        
        loc, dim, rot = cb.get_center(), cb.get_size(), abs(cb.get_rotation()) 
        pt_idx = self.select_pt(loc, rot)
        xi_2d, yi_2d, xi_close_2d, yi_close_2d = self.get_pts_2d()

        val_idxs = PT_IDX_DICT.get(str(pt_idx))
        ##//val: [ x_t, y_t, x_b, y_b, intercept_uppper, intercept_lower ]
        x_t, y_t = xi_2d[val_idxs[0] -1], yi_2d[val_idxs[1] -1]     #//cuboid 위쪽 내측 2d 좌표   
        x_b, y_b = xi_2d[val_idxs[2] -1], yi_2d[val_idxs[3] -1]     #//cuboid 아래쪽 외측 2d 좌표
        
        if semantical:
            # x_t = xi_2d[val_idxs[0] -1] #- coeff0*coeff1
            
            #// cuboid 위쪽 외측 선 - 카메라 외곽 intersection의 y좌표
            a_upper, b_upper = self.get_border_pt(xi_2d[val_idxs[4] -1], yi_2d[val_idxs[4] -1], xi_close_2d[val_idxs[4] -1], yi_close_2d[val_idxs[4] -1])
            if loc[1] > 0: y_t_corr = b_upper
            else: y_t_corr = 1920*a_upper + b_upper
            
            #// 작을 경우(이미지 상 위쪽)만 replace 
            if y_t_corr < y_t: y_t = y_t_corr
            
            #// 위와 같음 
            a_lower, b_lower = self.get_border_pt(xi_2d[val_idxs[5] -1], yi_2d[val_idxs[5] -1], xi_close_2d[val_idxs[5] -1], yi_close_2d[val_idxs[5] -1])
            if loc[1] > 0: y_b = b_lower
            else: y_b = 1920*a_lower + b_lower

        #//해상도 이내로 조정
        x_edge=1920
        if loc[1] > 0: x_edge = 0
        if y_t < 0 or y_t > 1200: y_t = 1200
        if y_b < 0 or y_b > 1200: y_b = 1200
        if x_t < 0 or x_t > 1920: x_t = x_edge
        if x_b < 0 or x_b > 1920: x_b = x_edge

        center_x, center_y =self.limit_val((x_t + x_b)/2, 1920), self.limit_val((y_t + y_b)/2, 1200)         
        w, h = self.limit_val(abs(x_t - x_b), 1920), self.limit_val(abs(y_t - y_b), 1200) 
        area_2d = self.limit_val(w*h, 1920*1200)
        box_2d = [center_x, center_y, w, h]
        return [box_2d, area_2d]
    
    def select_pt(self, loc, rot):
        if loc[1] > 0:
            if rot < 90: return 4
            else: return 1
        else:
            if rot < 90: return 3
            else: return 2
    
    def pt2d_outofbound(self, xi_2d, yi_2d):
        #2d 변환 좌표값 중 하나라도 카메라 좌표 밖일 시 True
        if any([x<0 or x>1920 for x in xi_2d]):
            return True
        if any([y<0 or y>1200 for y in yi_2d]):
            return True
        return False
    
    def mk_trigger(self):
        cb = self.get_cuboid()
        loc = cb.get_center()
        pts_arr = self.get_cuboid_pts()
        
        min_x_3d =min(list(np.array(pts_arr)[:,0]))
        fov_edge_x = 0.0
        loc_1 = loc[1]
        
        ##//val: [ coeff_ypos, intercept_ypos, coeff_yneg, intercept_yneg ]
        #// 3d 상 카메라 화각 경계 x좌표값 
        if loc_1>0: fov_edge_x = loc_1 / self.cam_fov_val[0] 
        else: fov_edge_x = loc_1 / self.cam_fov_val[2]
        
        #// cuboid x3d_min과의 차이값으로 경계 벗어남 판별
        interval = min_x_3d - fov_edge_x
        if interval < 0: return True
            
        """
        solve min_x_3d =min(list(np.array(pts_arr)[:,0]))
        
        init fov_edge_x
        case loc_y > 0
            fov_edge_x = loc_y / self.cam_fov_val[0]
        case loc_y < 0
            fov_edge_x = loc_y / self.cam_fov_val[2]
            
        interval = min_x_3d - fov_edge_x 
        if interval < CONSTANT
            return True 
        """
        """
        rot == 0
          3----4
         /|   /|
        1----2 |
        | 7--|-8
        |/   |/
        5----6

        rot == 180
          2----1
         /|   /|
        4----3 |
        | 6--|-5
        |/   |/
        8----7
        
        1: 4 
        2: 3
        3: 2
        4: 1
        
        pt[pt idx opposite] is out of bounds
        # if loc[0] < -10 or loc[0]-(dim[2]/2) > 6: return False
        # if loc[0] < -10 or loc[0]-(dim[2]/2) > 30: return False
        
        # pts_2d = self.get_pts_2d()
        # xi_2d = pts_2d[0]
        # yi_2d = pts_2d[1]
        # if self.pt2d_outofbound(xi_2d, yi_2d): return True
        """
        return False
    
    
    ##//======== calib calculation ========
    def zxy_to_xy(self, z, x, y, tvec, euler, K, D):
        fx, fy, cx, cy = K  #//cam intrinsic
        k1, k2, p1, p2 = D  #//cam intrinsic distortion
        xyz_e_mat = self.get_xyz_euler(x, -y, -z, tvec, euler) 
        xy_u_mat = self.get_xy_u(xyz_e_mat)
        sqr = self.get_sqr(xy_u_mat)
        xy_d_mat = self.get_xy_d(sqr, xy_u_mat, k1, k2, p1, p2)
        xy_p_mat = self.get_xy_p(xy_d_mat, fx, fy, cx, cy)
        
        if self.is_recent==False:
            return np.array([ [ xy_p_mat[0][0]-60, xy_p_mat[1][0] - 40 ]   ])
        else:
            return np.array([ [ xy_p_mat[0][0]-100, xy_p_mat[1][0] + 4 ]   ])
    
    def get_xyz_euler(self, x, y, z, tvec, euler):
        tx, ty, tz = tvec
        ud, lr, rot = euler
        op1 = np.asarray([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ])
        op2 = np.asarray([
            [1, 0, 0],
            [0, math.cos(rot), -math.sin(rot)],
            [0, math.sin(rot), math.cos(rot)]
        ])
        op3 = np.asarray([
            [math.cos(ud), 0, math.sin(ud)],
            [0, 1, 0],
            [-math.sin(ud), 0, math.cos(ud)]
        ])
        op4 = np.asarray([
            [math.cos(lr), -math.sin(lr), 0],
            [math.sin(lr), math.cos(lr), 0],
            [0,0, 1]
        ])
        op5 = np.asarray([
            [x - tx],
            [y - ty],
            [z - tz]
        ])

        ret = np.matmul(op1, op2)
        ret = np.matmul(ret, op3)
        ret = np.matmul(ret, op4)
        ret = np.matmul(ret, op5)
        return ret

    def get_xy_u(self, xyz_e_mat):
        xe = xyz_e_mat[0][0]
        ye = xyz_e_mat[1][0]
        ze = xyz_e_mat[2][0]
        if ze !=0: return np.asarray([[xe/ze],[ye/ze]])
        else: return np.asarray([[ 0 ],[ 0 ]])

    def get_sqr(self, xy_u_mat): ##// (2,1)
        xu = xy_u_mat[0][0]
        yu = xy_u_mat[1][0]
        return xu*xu + yu*yu 
        
    def get_xy_d(self, sqr, xy_u_mat, k1, k2, p1, p2):
        op1_coeff = 1 + k1*sqr + k2*sqr*sqr
        xu = xy_u_mat[0][0]
        yu = xy_u_mat[1][0]
        op1 = np.asarray([
            [xu * op1_coeff],
            [yu * op1_coeff]
        ])
        op2 = np.asarray([
            [2*p1*xu*yu + p2*(sqr + 2*xu*xu)],
            [p1*(sqr + 2*yu*yu) + 2*p2*xu*yu]
        ])
        return op1 + op2
        
    def get_xy_p(self, xy_d_mat, fx, fy, cx, cy):
        xd = xy_d_mat[0][0]
        yd = xy_d_mat[1][0]
        
        op1 = np.asarray([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 0]
        ])
        op2 = np.asarray([
            [xd],
            [yd],
            [1]
        ])
        ret = np.matmul(op1, op2)
        return np.asarray( [[ ret[0][0] ],[ ret[1][0] ]] )

    def parse_cal(self, calib):
        fx = float(calib['fx'])
        fy = float(calib['fy'])
        cx = float(calib['cx'])
        cy = float(calib['cy'])
        k1 = float(calib['k1'])
        k2 = float(calib['k2'])
        p1 = float(calib['p1/k3'])
        p2 = float(calib['p2/k4'])
        euler = calib["rvec"]
        tvec = calib['translation_vector']
        K = [fx, fy, cx, cy]
        D = [k1, k2, p1, p2]
        return K, D, euler, tvec
            
    def rotate_2d(self, target_pt, base_pt, theta, is_deg=True):
        ##// 2차원 평면에서 회전
        if is_deg == True:
            theta = theta*math.pi / 180
        x, y = target_pt[0], target_pt[1]
        base_x, base_y = base_pt[0], base_pt[1]
        ret_x = (x-base_x)*math.cos(theta) - (y-base_y)*math.sin(theta) + base_x
        ret_y = (x-base_x)*math.sin(theta) + (y-base_y)*math.cos(theta) + base_y
        ret_pt = np.array([ret_x, ret_y])
        return ret_pt
    

class Proj2dBox_A_d:
    def __init__(self, cuboid, cal_dict, cam_fov_val):
        self.cuboid = cuboid
        
        self.calib = cal_dict
        self.K, self.D, self.euler, self.tvec = self.parse_cal(self.calib)
        self.cam_fov_val = cam_fov_val
        
        self.cuboid_pts = self.mk_cuboid_pts()
        self.cuboid_close_pts = self.mk_cuboid_close_pts()
        self.pts_2d = self.mk_pts_2d()
        
    def set_cuboid(self, cuboid):
        self.cuboid = cuboid
    def get_cuboid(self):
        return self.cuboid
    def get_K(self):
        return self.K
    def get_D(self):
        return self.D
    def get_euler(self):
        return self.euler
    def get_tvec(self):
        return self.tvec
    def get_cuboid_pts(self):
        return self.cuboid_pts
    def get_cuboid_close_pts(self):
        return self.cuboid_close_pts
    def get_pts_2d(self):
        return self.pts_2d

    def mk_cuboid_pts(self):
        cb = self.get_cuboid()
        return cb.get_cuboid_pts()
    def mk_cuboid_close_pts(self):
        cb = self.get_cuboid()
        return cb.get_cuboid_pts_close()
    def mk_pts_2d(self):
        pts_arr = self.get_cuboid_pts()
        pts_close_arr = self.get_cuboid_close_pts()
        
        xi_2d = []
        yi_2d = []
        xi_close_2d = []
        yi_close_2d = []
        
        for pt, pt_close in zip(pts_arr, pts_close_arr):
            x, y, z = pt
            x_close, y_close, z_close = pt_close
            xy_2d = self.zxy_to_xy(z, x, y, self.get_tvec(), self.get_euler(), self.get_K(), self.get_D())
            xy_2d_close = self.zxy_to_xy(z_close, x_close, y_close, self.get_tvec(), self.get_euler(), self.get_K(), self.get_D())
            xi_2d.append(xy_2d[0][0])
            yi_2d.append(xy_2d[0][1])
            xi_close_2d.append(xy_2d_close[0][0])
            yi_close_2d.append(xy_2d_close[0][1])

        return [xi_2d, yi_2d, xi_close_2d, yi_close_2d]    
        
    def get_border_pt(self, x1, y1, x2, y2):
        a = 0
        b = 0
        if x1!=x2:
            a = (y2-y1)/(x2-x1)
            b = y1 - (a*x1)
        return a, b
    def limit_val(self, k, maxres):
        if k < 0: k = 0
        elif k>maxres: k = maxres
        return k
    
    
    def proj_2d_normal(self):
        """_summary_
        발산 보정 미적용 2d box 값  
        Returns:
            _list_: _[[x,y,w,h], area]_
        """        
        box_2d, area_2d = [], 0
        pts_2d = self.get_pts_2d()
        xi_2d = pts_2d[0]
        yi_2d = pts_2d[1]
        xmin, xmax, ymin, ymax = self.limit_val(min(xi_2d), 1920), self.limit_val(max(xi_2d), 1920), \
                                 self.limit_val(min(yi_2d), 1200), self.limit_val(max(yi_2d), 1200)
        center_x, center_y, w, h = (xmax+xmin)/2, (ymax+ymin)/2, abs(xmax-xmin), abs(ymax-ymin)
        box_2d = [center_x, center_y, w, h]
        area_2d = w*h
        return [box_2d, area_2d]

    def correct_2d_indev(self, semantical=False):
        """_summary_

        Args:
            semantical (bool, optional): _True:근거리 의미적 보정 적용; False:단순 발산 보정만 적용_. Defaults to False.

        Returns:
            _list_: _[[x,y,w,h], area]_
        """        
        cb = self.get_cuboid()
        
        loc, dim, rot = cb.get_center(), cb.get_size(), abs(cb.get_rotation())
        pt_idx = self.select_pt(loc, rot)
        xi_2d, yi_2d, xi_close_2d, yi_close_2d = self.get_pts_2d()

        # x2d_pivot = 10
        # trg_val = loc[0]+(dim[2]/2)
        # coeff0 = x2d_pivot-trg_val
        # coeff1 = 30

        val_idxs = PT_IDX_DICT.get(str(pt_idx))
        ##//val: [ x_t, y_t, x_b, y_b, intercept_uppper, intercept_lower ]
        x_t, y_t = xi_2d[val_idxs[0] -1], yi_2d[val_idxs[1] -1]
        x_b, y_b = xi_2d[val_idxs[2] -1], yi_2d[val_idxs[3] -1]
        if semantical:
            x_t = xi_2d[val_idxs[0] -1] #- coeff0*coeff1
            a_upper, b_upper = self.get_border_pt(xi_2d[val_idxs[4] -1], yi_2d[val_idxs[4] -1], xi_close_2d[val_idxs[4] -1], yi_close_2d[val_idxs[4] -1])
            a_lower, b_lower = self.get_border_pt(xi_2d[val_idxs[5] -1], yi_2d[val_idxs[5] -1], xi_close_2d[val_idxs[5] -1], yi_close_2d[val_idxs[5] -1])
            if loc[1] > 0: 
                y_t = b_upper
                y_b = b_lower
            else:
                y_t = 1920*a_upper + b_upper
                y_b = 1920*a_lower + b_lower
        x_edge=1920
        if loc[1] > 0: x_edge = 0
        if y_t < 0 or y_t > 1200: y_t = 1200
        if y_b < 0 or y_b > 1200: y_b = 1200
        if x_t < 0 or x_t > 1920: x_t = x_edge
        if x_b < 0 or x_b > 1920: x_b = x_edge

        center_x, center_y = self.limit_val((x_t + x_b)/2, 1920), self.limit_val((y_t + y_b)/2, 1200)         
        w, h = self.limit_val(abs(x_t - x_b), 1920), self.limit_val(abs(y_t - y_b), 1200) 
        area_2d =self.limit_val(w*h, 1920*1200)
        box_2d = [center_x, center_y, w, h]
        return [box_2d, area_2d]
        
    
    def select_pt(self, loc, rot):
        if loc[1] > 0:
            if rot < 90: return 4
            else: return 1
        else:
            if rot < 90: return 3
            else: return 2
    
    def pt2d_outofbound(self, xi_2d, yi_2d):
        if any([x<0 or x>1920 for x in xi_2d]): return True
        if any([y<0 or y>1200 for y in yi_2d]): return True
        return False
        
    def mk_trigger(self):
        cb = self.get_cuboid()
        loc = cb.get_center()
        pts_arr = self.get_cuboid_pts()
        
        min_x_3d =min(list(np.array(pts_arr)[:,0]))
        fov_edge_x = 0.0
        loc_1 = loc[1]
        if loc_1>0: fov_edge_x = loc_1 / self.cam_fov_val[0] 
        else: fov_edge_x = loc_1 / self.cam_fov_val[2]
        interval = min_x_3d - fov_edge_x
        if interval < 0: return True

        return False
    
    
    
    ##//======== calib calculation ========
    def zxy_to_xy(self, z, x, y, tvec, euler, K, D):
        fx, fy, cx, cy = K
        k1, k2, p1, p2 = D
        
        xyz_e_mat = self.get_xyz_euler(x, -z, y, tvec, euler)
        xy_u_mat = self.get_xy_u(xyz_e_mat)
        sqr = self.get_sqr(xy_u_mat)
        xy_d_mat = self.get_xy_d(sqr, xy_u_mat, k1, k2, p1, p2)
        xy_p_mat = self.get_xy_p(xy_d_mat, fx, fy, cx, cy)
        
        ret_pt = self.rotate_2d([xy_p_mat[0][0], xy_p_mat[1][0]], [960,600], -90)
        
        return np.array([[ ret_pt[0]-8, ret_pt[1]-32 ]])
        
    def get_xyz_euler(self, x, y, z, tvec, euler):
         
        tx, tz, ty = tvec
        lr, ud, rot = euler
        
        op1 = np.asarray([
            [0, 1, 0],
            [0, 0, -1],
            [-1, 0, 0]
        ])
        op2 = np.asarray([
            [1, 0, 0],
            [0, math.cos(rot), -math.sin(rot)],
            [0, math.sin(rot), math.cos(rot)]
        ])
        op3 = np.asarray([
            [math.cos(ud), 0, math.sin(ud)],
            [0, 1, 0],
            [-math.sin(ud), 0, math.cos(ud)]
        ])
        op4 = np.asarray([
            [math.cos(lr), -math.sin(lr), 0],
            [math.sin(lr), math.cos(lr), 0],
            [0,0, 1]
        ])
        op5 = np.asarray([
            [x - tx],
            [y - ty],
            [z + tz]
        ])
       
        ret = np.matmul(op1, op2)
        ret = np.matmul(ret, op3)
        ret = np.matmul(ret, op4)
        ret = np.matmul(ret, op5)
        return ret

    def get_xy_u(self, xyz_e_mat):
        xe = xyz_e_mat[0][0]
        ye = xyz_e_mat[1][0]
        ze = xyz_e_mat[2][0]
        if ze !=0: return np.asarray([[xe/ze],[ye/ze]])
        else: return np.asarray([[ 0 ],[ 0 ]])

    def get_sqr(self, xy_u_mat): ##// (2,1)
        xu = xy_u_mat[0][0]
        yu = xy_u_mat[1][0]
        return xu*xu + yu*yu 
        
    def get_xy_d(self, sqr, xy_u_mat, k1, k2, p1, p2):
        op1_coeff = 1 + k1*sqr + k2*sqr*sqr
        xu = xy_u_mat[0][0]
        yu = xy_u_mat[1][0]
        op1 = np.asarray([
            [xu * op1_coeff],
            [yu * op1_coeff]
        ])
        op2 = np.asarray([
            [2*p1*xu*yu + p2*(sqr + 2*xu*xu)],
            [p1*(sqr + 2*yu*yu) + 2*p2*xu*yu]
        ])
        return op1 + op2
        
    def get_xy_p(self, xy_d_mat, fx, fy, cx, cy):
        xd = xy_d_mat[0][0]
        yd = xy_d_mat[1][0]
        
        op1 = np.asarray([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 0]
        ])
        op2 = np.asarray([
            [xd],
            [yd],
            [1]
        ])
        
        ret = np.matmul(op1, op2)
        
        return np.asarray( [[ ret[0][0] ],[ ret[1][0] ]] )

    def parse_cal(self, calib):
        fx = float(calib['fx'])
        fy = float(calib['fy'])
        cx = float(calib['cx'])
        cy = float(calib['cy'])
        k1 = float(calib['k1'])
        k2 = float(calib['k2'])
        p1 = float(calib['p1/k3'])
        p2 = float(calib['p2/k4'])
        euler = calib["rvec"]
        tvec = calib['translation_vector']
        K = [fx, fy, cx, cy]
        D = [k1, k2, p1, p2]
        
        return K, D, euler, tvec

    def rotate_2d(self, target_pt, base_pt, theta, is_deg=True):
        ##// 2차원 평면에서 회전
        if is_deg == True:
            theta = theta*math.pi / 180
        x, y = target_pt[0], target_pt[1]
        base_x, base_y = base_pt[0], base_pt[1]
        ret_x = (x-base_x)*math.cos(theta) - (y-base_y)*math.sin(theta) + base_x
        ret_y = (x-base_x)*math.sin(theta) + (y-base_y)*math.cos(theta) + base_y
        ret_pt = np.array([ret_x, ret_y])
        return ret_pt
    

#// polygon 보정 class
class PlgCorr:
    def __init__(self, box_arr, multiplg_arr, clipname):
        self.box_arr = box_arr
        self.multiplg_arr = multiplg_arr
        self.clipname = clipname
        self.plg_arr = None
    
    def get_clipname(self):
        return self.clipname
    def get_plg_arr(self):
        return self.plg_arr
    def set_plg_arr(self, plg_arr):
        self.plg_arr = plg_arr
    
    def applicate_corr(self):
        corr_flag = False
        box_arr = self.box_arr
        multiplg_arr = self.multiplg_arr
        try:
            box_3d = self.select_3dbox(box_arr) #// 가장 가까운 cuboid 선택
            loc, dim, rot_y = box_3d["location"], box_3d["dimension"], box_3d["rotation_y"]
            cb = Cuboid(loc, dim, rot_y)
            lstr_hrz, lstr_vrt, lstr_hrz_lim, lstr_vrt_lim = self.cuboid_linestring(cb) #// cuboid front 2dbox: 상측, 내측 LineString instance
            pivot_lstr = lstr_vrt

            #// cuboid front 2dbox 모양에 따라 상측 / 내측 LineString 선택 
            shape_vt = True
            if lstr_hrz_lim.length > lstr_vrt_lim.length: 
                pivot_lstr = lstr_hrz
                shape_vt = False
            # flag = "hr","vr"
            
            #// cuboid 3d 좌/우측 위치 
            loc_left = True
            if loc[1]<0: loc_left = False
            
            
            plg_arr, plg_cent, plg_idx = self.select_plg(loc_left, multiplg_arr)    #// 보정 대상 폴리곤 선택 (가장 내측 위치)
            plg_instance, plg_arr = self.ordered_plg_instance(plg_arr, plg_cent)    #// self-intersection solve 
            self.set_plg_arr(plg_arr)
            
            intersect_pt_arr = self.intersect_lstr_plg(pivot_lstr, plg_instance)    #// LineString - polygon intersection 
            
            # rm_intersect_idx_arr = self.undersized_intersect_idxs(loc_left, shape_vt, intersect_pt_arr, pivot_lstr)
            # for index in sorted(rm_intersect_idx_arr, reverse=True): del intersect_pt_arr[index]
            
            if intersect_pt_arr: 
                rm_idx_arr = self.oversized_pt_idxs(loc_left, shape_vt, plg_arr, pivot_lstr)    #// pivot LineString의 경계 초과한 pt의 idx 구하기    
                for index in sorted(rm_idx_arr, reverse=True): del plg_arr[index]               #// 경계 초과 pt 삭제
                plg_arr = plg_arr + intersect_pt_arr                                            #// intersection pt 추가 
            
                #// polygon arr reordering
                plg_cent = self.calc_centroid(plg_arr)                                          
                plg_instance, plg_arr = self.ordered_plg_instance(plg_arr, plg_cent)            
                self.set_plg_arr(plg_arr)
                
                corr_flag = True
            
            return self.get_plg_arr(), plg_idx, corr_flag
        
        except Exception:
            #// 예외 시 보정 미실시
            return self.get_plg_arr(), -1, False
        
        
    def select_3dbox(self, box_arr):
        ret_box = box_arr[0]
        for box in box_arr:
            x3d,y3d,z3d = box["location"]
            x3d_ret,y3d_ret,z3d_ret = ret_box["location"]
            if x3d>x3d_ret: ret_box = box
        return ret_box
    
    def cuboid_linestring(self, cuboid):
        loc, dim, rot_y = cuboid.get_center(), cuboid.get_size(), cuboid.get_rotation()
        pt_idx = self.slv_pt_idx(loc[1], rot_y)
        val_idxs = PT_IDX_DICT.get(str(pt_idx))
        pt_idx_hrz, pt_idx_vrt = val_idxs[-2], val_idxs[-1]
        p2b = Proj_2d_Wrapper(self.get_clipname(), cuboid).proj_module
        xi_2d, yi_2d, _, _ = p2b.get_pts_2d()
        
        core_pt = (xi_2d[pt_idx - 1], yi_2d[pt_idx - 1])

        lstr_hrz = shapely.LineString([[0.0, core_pt[1]], [1920.0, core_pt[1]]]) #.length
        lstr_vrt = shapely.LineString([[core_pt[0], 0.0], [core_pt[0], 1200.0]])
        
        pt_hrz = (xi_2d[pt_idx_hrz - 1], yi_2d[pt_idx_hrz - 1])
        pt_vrt = (xi_2d[pt_idx_vrt - 1], yi_2d[pt_idx_vrt - 1])
        lstr_hrz_lim = shapely.LineString([core_pt, pt_hrz]) #.length
        lstr_vrt_lim = shapely.LineString([core_pt, pt_vrt])
        
        return lstr_hrz, lstr_vrt, lstr_hrz_lim, lstr_vrt_lim
    
    def select_plg(self, loc_left, multiplg_arr):
        ret_plg = multiplg_arr[0]
        ret_plg_cent = self.calc_centroid(ret_plg)
        ret_idx = 0
        for i, plg in enumerate(multiplg_arr):
            plg_cent = self.calc_centroid(plg)
            if loc_left:
                if plg_cent[0] > ret_plg_cent[0]: # centeroid_x is larger 
                    ret_plg = plg
                    ret_plg_cent = plg_cent
                    ret_idx = i 
            else:
                if plg_cent[0] < ret_plg_cent[0]: # centeroid_x is smaller 
                    ret_plg = plg 
                    ret_plg_cent = plg_cent
                    ret_idx = i
        return ret_plg, ret_plg_cent, ret_idx        
        
        
    
        
    def intersect_lstr_plg(self, lstr, plg):
        if plg.is_simple and plg.intersects(lstr):
            try:
                intersect_mpts = plg.intersection(lstr).boundary.geoms
                intersect_pt_arr = [ list(*pt.coords) for pt in intersect_mpts]
                return intersect_pt_arr
            except RuntimeWarning:
                return []
        return []
    
    def oversized_pt_idxs(self, loc_left, shape_vt, plg_arr, lstr):
        lx,ly = lstr.coords.xy
        lx,ly = lx.tolist(), ly.tolist()
        xmin, xmax, ymin, ymax = min(lx), max(lx), min(ly), max(ly)
        rm_idx_arr = []
        for i, xy in enumerate(plg_arr):
            x, y = xy
            if shape_vt:
                if loc_left:
                    pivot = xmax
                    if x>pivot: rm_idx_arr.append(i)
                else:
                    pivot = xmin
                    if x<pivot: rm_idx_arr.append(i)
            else:
                pivot = ymin
                if y<pivot: rm_idx_arr.append(i)
        return rm_idx_arr
    
    def undersized_intersect_idxs(self, loc_left, shape_vt, intersect_pt_arr, lstr):
        lx,ly = lstr.coords.xy
        lx,ly = lx.tolist(), ly.tolist()
        xmin, xmax, ymin, ymax = min(lx), max(lx), min(ly), max(ly)
        rm_idx_arr = []
        for i, xy in enumerate(intersect_pt_arr):
            x, y = xy
            if shape_vt:
                if loc_left:
                    pivot = xmax
                    if x<pivot: rm_idx_arr.append(i)
                else:
                    pivot = xmin
                    if x>pivot: rm_idx_arr.append(i)
            else:
                pivot = ymin
                if y>pivot: rm_idx_arr.append(i)
        return rm_idx_arr
    
    def add_intersect(self):
        return
    
    def ordered_plg_instance(self, plg_arr, plg_cent):
        plg = shapely.Polygon(plg_arr)
        if plg.is_simple==False:
            plg_arr.sort(key=lambda p: math.atan2(p[1]-plg_cent[1],p[0]-plg_cent[0]), reverse=True) 
            plg = shapely.Polygon(plg_arr)
        return plg, plg_arr
    
    def calc_centroid(self, plg_arr):
        return [sum([p[0] for p in plg_arr])/len(plg_arr),sum([p[1] for p in plg_arr])/len(plg_arr)]
    
    def slv_pt_idx(self, loc_1, rot_y):
        if loc_1 > 0:
            if abs(rot_y) < 90: return 4
            else: return 1
        else:
            if abs(rot_y) < 90: return 3
            else: return 2




"""

    def calc_loc_trans_at_origin_legacy(self, center, size, rmat, fit_target, category, min_vals, max_vals):
        
        fit_center = [0.0, 0.0, 0.0]    # set origin 
        fit_size = size                 # default dim
        

        # min max vals of inbox pcd
        xmin, ymin, zmin = min_vals     
        xmax, ymax, zmax = max_vals

        if 'w' in fit_target or 'y' in fit_target: #// fit y-axis (width)
            fit_size[0] = abs(ymin-ymax)
            fit_center[1] = (ymin+ymax)/2

        if 'h' in fit_target or 'z' in fit_target: #// fit z-axis (height)
            fit_size[1] = abs(zmin-zmax)
            fit_center[2] = (zmin+zmax)/2
            
        if 'l' in fit_target or 'x' in fit_target: #// fit x-axis (length)
            fit_size[2] = abs(xmin-xmax)
            fit_center[0] = (xmin+xmax)/2

        #// reshape
        tmp_pt = np.array([ fit_center ]).reshape((3,1))
        
        #// rotate at origin
        pt_r = (rmat @ tmp_pt).reshape((3,))

        #// restore from temp translation
        pt_r = pt_r + np.array(center).reshape((3,))

        fit_center = pt_r.tolist()
        return fit_center, fit_size

"""