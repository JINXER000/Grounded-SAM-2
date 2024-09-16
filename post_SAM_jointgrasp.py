
import numpy as np
import os
SAM_PATH = os.getcwd()
import cv2
import json
import h5py

from aloha_segmenter import ObjectTrackingPipeline

from tqdm import tqdm

import sys
TAMP_PATH = '/home/xuhang/interbotix_ws/src/pddlstream_aloha/'
sys.path.append(TAMP_PATH)
os.chdir(TAMP_PATH)
from examples.pybullet.aloha_real.openworld_aloha.estimation.dnn import iterate_array
from examples.pybullet.aloha_real.openworld_aloha.policy_simp import get_compatible_campose
from examples.pybullet.aloha_real.openworld_aloha.entities import  CameraImage
from examples.pybullet.aloha_real.openworld_aloha.estimation.observation import iterate_image, custom_iterate_point_cloud
from examples.pybullet.aloha_real.insertion_gmm.post_jointgrasp import get_start_end_objs, play_vid, _smooth,\
    save_dict_to_hdf5, qpos_to_eetrans, PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN, DT
from examples.pybullet.aloha_real.openworld_aloha.simple_worlds import render_pose
from examples.pybullet.utils.pybullet_tools.utils import aabb_from_points, get_aabb_extent


os.chdir(SAM_PATH)

def compute_eepath(joint_val_list, robot_id):

    ee_path = [ qpos_to_eetrans(joint_val, robot_id) for joint_val in joint_val_list]
    return ee_path


def filter_grasp_by_obj(obj_center, ee_path, grasp_ids, threshold = 0.1):
    filtered_grasp_ids = []

    for idx in grasp_ids:
        eetrans_t = ee_path[idx][:3, 3].reshape(3)

        grasp_dist = np.linalg.norm(eetrans_t - obj_center)  
        if grasp_dist < threshold:
            filtered_grasp_ids.append(idx)

    return filtered_grasp_ids

def filter_grasp_starting(grasp_ids, start_id=80):
    filtered_grasp_ids = [grasp_id for grasp_id in grasp_ids if grasp_id > start_id]
    return filtered_grasp_ids

def filter_release_by_seq(grasp_ids, release_ids):
    filtered_grasp_ids = [release_id for release_id in release_ids if release_id > grasp_ids[-1]]
    return filtered_grasp_ids

def downsample_pc(pc, target_size = 512):
    pc = np.array(pc)

    ### save the point cloud as pcd
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)

    # outlier removal
    # cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=0.2)
    cl, ind = pcd.remove_radius_outlier(nb_points=40, radius=0.05)
    cl_pts = np.asarray(cl.points)
    cl_pts = cl_pts[cl_pts[:, 2] > 0.03]

    # obj_aabb = aabb_from_points(cl_pts)
    # obj_extent = get_aabb_extent(obj_aabb)
    # if obj_extent[2] > 0.15:
    #     o3d.io.write_point_cloud("dbg.ply", cl)
    #     print('debug')

    if len(cl_pts) < target_size:
        raise NotImplementedError("Not enough points in the point cloud")
    sampled_indices = np.random.choice(cl_pts.shape[0], target_size, replace=False)
    out_pc = cl_pts[sampled_indices]


    return out_pc

def actions2grasps(obs_cam_high, obs_high_depth, camera_info, cam_pose_json_file, \
                                        video_segments, objects, actions, target_size = 512, objs = None,  plot = False):
    l_joint_vals = actions[:, :6]
    l_gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(actions[:, 6])
    r_joint_vals = actions[:, 7:13]
    r_gripper_val = PUPPET_GRIPPER_JOINT_UNNORMALIZE_FN(actions[:, 13])

    smoothed_gripper_ql = _smooth(l_gripper_val)
    gripper_dql = np.diff(smoothed_gripper_ql, axis=0) / DT

    l_grasp_ids = switch_ids(smoothed_gripper_ql, gripper_dql, x_threshold=0.35, dx_threshold=-0.7, type='grasp')
    l_release_ids = switch_ids(smoothed_gripper_ql, gripper_dql, x_threshold=0.6, dx_threshold= 0.7, clip_start=100, type='release')

    smoothed_gripper_qr = _smooth(r_gripper_val)
    gripper_dqr = np.diff(smoothed_gripper_qr, axis=0) / DT

    r_grasp_ids = switch_ids(smoothed_gripper_qr, gripper_dqr, x_threshold=0.35, dx_threshold=-0.7, type='grasp')
    r_release_ids = switch_ids(smoothed_gripper_qr, gripper_dqr, x_threshold=0.6, dx_threshold= 0.7, clip_start=100, type='release')

    left_eepath = compute_eepath(l_joint_vals, 0)
    right_eepath = compute_eepath(r_joint_vals, 1)
    # filter the ids using the distance to the object

    start_obj_points = [lp.point for lp in objs[0].points]
    start_obj_points = downsample_pc(start_obj_points)
    end_obj_points = [lp.point for lp in objs[-1].points]     
    end_obj_points = downsample_pc(end_obj_points)

    l_grasp_ids = filter_grasp_starting(l_grasp_ids)
    r_grasp_ids = filter_grasp_starting(r_grasp_ids)
    l_release_ids = filter_release_by_seq(l_grasp_ids, l_release_ids)
    r_release_ids = filter_release_by_seq(r_grasp_ids, r_release_ids)
 
    if plot:
        import matplotlib.pyplot as plt
        plt.clf()
        plt.plot(smoothed_gripper_ql, label='smoothed_left_qpos')
        plt.plot(smoothed_gripper_qr, label='smoothed_right_qpos')

        # draw grasp and release points
        plt.scatter(l_grasp_ids, smoothed_gripper_ql[l_grasp_ids], c='r', label='left_grasp')
        plt.scatter(l_release_ids, smoothed_gripper_ql[l_release_ids], c='g', label='left_release')
        plt.scatter(r_grasp_ids, smoothed_gripper_qr[r_grasp_ids], c='b', label='right_grasp')
        plt.scatter(r_release_ids, smoothed_gripper_qr[r_release_ids], c='y', label='right_release')
        plt.legend()
        pic_name = 'gripper_vals.png'
        plt.savefig(pic_name)        


    moving_ids = list(range(min(r_grasp_ids), max(r_release_ids)))
    
    demo_joint_vals = [actions[i] for i in moving_ids]
    # pred_joint_data = zip(moving_ids, demo_joint_vals)

    r_grasping_poses = [right_eepath[i] for i in moving_ids]
    pc_moving = []
    for i in tqdm(moving_ids):
        # if i == 158:
        #     print('debug')
        obj_pc = get_labeled_points(obs_cam_high[i], obs_high_depth[i], camera_info, cam_pose_json_file, \
                                            video_segments[i][1][0], objects)
        pc_moving.append(obj_pc)

    r_grasp_poses = [right_eepath[i] for i in r_grasp_ids]
    pred_grasps =  r_grasp_poses+ r_grasping_poses
    pred_pcs = [start_obj_points]*len(r_grasp_ids) + pc_moving

    l_release_grasps = [left_eepath[i] for i in l_release_ids]

    contact_info_dict = {'pred_grasps': pred_grasps, 'pred_pcs': pred_pcs, \
                         'pred_ids': moving_ids, \
                         'eff_grasps': l_release_grasps, 'eff_pc': end_obj_points, \
                         'pred_joint_vals': demo_joint_vals}
    return contact_info_dict

def switch_ids(smoothed_qpos, gripper_change_rate, x_threshold = 0.1,  dx_threshold = -1, type = 'grasp', clip_start = 0):
    # gripper_data = qpos_data[:, -1]
    # smoothed_qpos = _smooth(gripper_data, smooth_window_size)
    # gripper_change_rate = np.diff(smoothed_qpos) / DT
    ret_ids = []
    if type == 'grasp':
        abrupt_window_size = 10
        for id in range(clip_start, len(gripper_change_rate)-abrupt_window_size+1):
            dx_window = gripper_change_rate[id:id+abrupt_window_size]
            x_window = smoothed_qpos[id:id+abrupt_window_size]
            if  np.min(dx_window)< dx_threshold:
                if x_window[-1] < x_threshold:
                    ret_ids.append(id)

    elif type == 'release':
        abrupt_window_size = 10
        for id in range(clip_start, len(gripper_change_rate)-abrupt_window_size+1):
            dx_window = gripper_change_rate[id:id+abrupt_window_size]
            x_window = smoothed_qpos[id:id+abrupt_window_size]
            if  np.max(dx_window)> dx_threshold:
                if x_window[-1] > x_threshold:
                    ret_ids.append(id)

    return ret_ids


# def get_labeled_points(color_img, depth_img_mm, camera_info_depth, cam_pose_json_file, \
#                        seg_mask, objects, max_depth=2.0, target_size = 512):
    
#     str_seg = np.full(depth_img_mm.shape + (2,), "unknown", dtype=object)
#     for r, c in iterate_array(str_seg, dims=[0, 1]):
#         if seg_mask[r, c] != False:
#             str_seg[r, c, 0] = objects[0]
#             str_seg[r, c, 1] = "instance_1"


#     # get the camera pose
#     assert cam_pose_json_file is not None
#     with open(cam_pose_json_file) as f:
#         cam_pose_dict = json.load(f)
        
#     camera_pose = get_compatible_campose(cam_pose_dict)
#     depth_camera_matrix = np.array(camera_info_depth['K']).reshape(3, 3)
#     depth_img = depth_img_mm.astype(np.float32) / 1000.0
#     camera_image = CameraImage(color_img, depth_img, str_seg, camera_pose, depth_camera_matrix)

#     ## convert labeled_image to labeled_points
#     pixels = [
#         pixel
#         for pixel in iterate_image(camera_image, step_size=2)
#         if camera_image.segmentationMaskBuffer[pixel][0] ==objects[0]
#     ]

#     relevant_cloud = [
#         lp.point
#         for lp in custom_iterate_point_cloud(
#             camera_image, pixels, max_depth=max_depth
#         )
#     ]

#     relevant_cloud = downsample_pc(relevant_cloud)

#     return relevant_cloud

def get_labeled_points(color_img, depth_img_mm, camera_info_depth, cam_pose_json_file, \
                          seg_mask, objects, max_depth=2.0, target_size = 512):
    
    depth_camera_matrix = np.array(camera_info_depth['K']).reshape(3, 3)
    fx = depth_camera_matrix[0, 0]
    fy = depth_camera_matrix[1, 1]
    cx = depth_camera_matrix[0, 2]
    cy = depth_camera_matrix[1, 2]

    y_indices, x_indices = np.where(seg_mask)

    depth_img = depth_img_mm.astype(np.float32) / 1000.0
    depth_values = depth_img[y_indices, x_indices]

    # Apply depth filtering: exclude values less than 0.3m and greater than 2m
    valid_depth_mask = (depth_values >= 0.3) & (depth_values <= 2.0)
    # Filter indices and depth values based on the valid depth mask
    y_indices = y_indices[valid_depth_mask]
    x_indices = x_indices[valid_depth_mask]
    depth_values = depth_values[valid_depth_mask]

    x_cam = (x_indices - cx) * depth_values / fx
    y_cam = (y_indices - cy) * depth_values / fy
    z_cam = depth_values

    points_camera = np.vstack([x_cam, y_cam, z_cam, np.ones_like(z_cam)])

    with open(cam_pose_json_file) as f:
        cam_pose_dict = json.load(f)

    xfront_cam_pose = [cam_pose_dict['xyz'], cam_pose_dict['wxyz']]
    # qw, qx, qy, qz = xfront_cam_pose[1]
    # quat_xyzw = [qx, qy, qz, qw]

    # cam_pose_T = np.eye(4)
    # import pybullet as p
    # cam_pose_T[:3, :3] = np.array(p.getMatrixFromQuaternion(quat_xyzw)).reshape(3, 3)
    # cam_pose_T[:3, 3] = xfront_cam_pose[0]

    cam_pose_7d = get_compatible_campose(cam_pose_dict)
    cam_pose_T = np.eye(4)
    import pybullet as p
    cam_pose_T[:3, :3] = np.array(p.getMatrixFromQuaternion(cam_pose_7d[1])).reshape(3, 3)
    cam_pose_T[:3, 3] = cam_pose_7d[0]

    points_world = np.dot(cam_pose_T, points_camera)

    points_Nx3 = downsample_pc(points_world[:3].T, target_size)



    return points_Nx3
    



def postprocess_jointgrasp(hdf5_path, play_orig_vid=False, cam_pose_json_file = None):
    data_dict = read_demo_realsense(hdf5_path)
    obs_qpos = data_dict['qpos']
    obs_cam_high = data_dict['cam_high']
    obs_high_depth = data_dict['high_depth']
    color_imgs = data_dict['color_img']
    depth_imgs = data_dict['depth_img']
    camera_info = json.loads(data_dict['camera_info'])

    if play_orig_vid:
        play_vid(obs_cam_high)    

    ### SAM on color image to produce segmentation mask
    rgb_segmenter = ObjectTrackingPipeline(text_prompt = 'tape.', source_video_frame_dir = "./custom_video_frames", \
                                          save_tracking_results_dir = "./tracking_results" )
    
    rgb_segmenter.save_images_frames(obs_cam_high)
    video_segments, objects = rgb_segmenter.run_pipeline()

    

    # # NOTE: if starting at perception state, we only have to collect end point cloud
    os.chdir(TAMP_PATH)
    start_obj, end_obj = get_start_end_objs(color_imgs, depth_imgs, camera_info, cam_pose_json_file)

    # identify grasp poses
    contact_info_dict = actions2grasps(obs_cam_high, obs_high_depth, camera_info, cam_pose_json_file, \
                                        video_segments, objects, actions=obs_qpos, objs = [start_obj, end_obj], plot = True
                                       )


    # save the contact info dict to the hdf5 file
    save_dict_to_hdf5(contact_info_dict, hdf5_path)
    print(f"------Processed  {hdf5_path.split('/')[-1]}----------")



    os.chdir(SAM_PATH)






def read_demo_realsense(input_hdf5_path, from_sim=False):
    # ensure path exist
    if not os.path.exists(input_hdf5_path):
        raise NameError("File not found!")
    with h5py.File(input_hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))
        keys = list(f.keys())

        data_dict = {}

        print("\n--- Action --- ")

        action = f['action'][()]
        print("action.shape: {}".format(action.shape))

        obs_grp = f['observations']
        obs_keys = list(obs_grp.keys())


        print(f"\nObservation keys: {obs_keys}")

        print("\n--- Joint --- ")

        # obs_efforts = obs_grp['effort'][()]
        obs_qpos = obs_grp['qpos'][()]
        obs_qvel = obs_grp['qvel'][()]

        # print("joint_efforts.shape: {}".format(obs_efforts.shape))
        print("joint_qpos.shape: {}".format(obs_qpos.shape))
        print("joint_qvel.shape: {}".format(obs_qvel.shape))

        data_dict['qpos'] = obs_qpos

        print("\n--- Observations --- ")

        obs_images = obs_grp['images']
        obs_image_keys = list(obs_images.keys())
        print(f"obs_image_keys: {obs_image_keys}")

        if from_sim:
            obs_cam_high = obs_images['top'][()]
        else:
            obs_cam_high = obs_images['cam_high'][()]

        data_dict['cam_high'] = obs_cam_high
        data_dict['high_depth'] = obs_images['high_depth'][()]

        print("obs_cam_high.shape: {}".format(obs_cam_high.shape))

        # get color and depth
        data_dict['color_img'] = f['color_img'][()]
        data_dict['depth_img'] = f['depth_img'][()]
        data_dict['camera_info'] = f['camera_info'][()]
        
        return data_dict

def visualize_processed(hdf5_path, is_pred = True):
    with h5py.File(hdf5_path, "r") as f:
        print("Keys: {}".format(f.keys()))
        if is_pred:
            grasps = f['pred_grasps'][()]
            tmp_pc = f['pred_pcs'][()]
            
        else:
            grasps = f['eff_grasps'][()]
            tmp_pc = f['eff_pc'][()]

    history_list = []
    for i in range(len(grasps)):
        action_slice = (grasps[i], None)
        history_list.append(action_slice)

    if is_pred:
        render_pose(history_list, use_gui=True, \
                directory = None, moving_pc_list = tmp_pc)
    else:
        render_pose(history_list, use_gui=True, \
                    directory = None, obj_points = tmp_pc)
    
    print(f"------Visualized {hdf5_path.split('/')[-1]}------")

if __name__ == '__main__':
    cam_pose_json_file = '/home/xuhang/interbotix_ws/src/pddlstream_aloha/examples/pybullet/aloha_real/openworld_aloha/estimation/temp_vis/camera_pose.json'
    
    # file_dir = "/home/xuhang/Desktop/aloha_data/aloha_transfer_tape"
    file_dir = "/ssd1/aloha_data/aloha_transfer_tape"
    # file_dir = "/home/xuhang/Desktop/yzchen_ws/equibot_abstract/data/transfer_tape/raw"
    
    for episode_idx in [9]:
    # for episode_idx in range(8, 10): 
        test_hdf5_path = os.path.join(file_dir, f"episode_{episode_idx}.hdf5")
        postprocess_jointgrasp(test_hdf5_path, play_orig_vid=False, \
                               cam_pose_json_file = cam_pose_json_file)
        # visualize_processed(test_hdf5_path, is_pred = True)