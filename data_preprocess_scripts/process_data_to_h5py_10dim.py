import numpy as np
import os
from scipy.spatial.transform import Rotation
import h5py
import cv2
# import transform_utils
import copy
import sys
# sys.path.append("/mnt/sda1/act") 
# make the current NumPy's core visible under the old name
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

"""
Convert the demos from Deoxys to hdf5 format used by ACT
Note that we use m as the unit rather than mm, so translation is divided by 1000
"""
# ----------------------- !!! Setting !!! -------------------- #
input_path = 'data/rebar_insertion' 
output_path = "data/rebar_insertion_hdf5_10dim" 
max_timesteps = 300 # # maximum timesteps for each episode
action_type = "abs" # "abs" for absolute position, "delta" for relative position
img_only = False # if we only use the image without the robot state, we need to set this to True
camera_names = ["webcam_1","webcam_2"]   # sequence can not be wrong
cam_idxes = ["1","2"] # "0","1","2"   sequence can not be wrong ["2","3","4"]
crop_wo_sam2_option = True  # if we crop the image, we need to set this to True
image_h = 700 # 900 # desired size 240 # 320 # 480 # 720 # 480 # 1080
image_w = 520 # 960 # desired size 320 # 480 # 640 # 1280 # 640 # 1920
crop_list = [[700, 200, 1220, 900]] # w1,h1,w2,h2

# ----------------------- !!! Advanced Setting !!! ----------------------- #
hdf5_start_idx = 0 # if we want to continue from the last hdf5 file, we can set this to the last hdf5 file index
# if we use sam2 to segment the image, we need to adjust the following parameters, we don't use them in the Ufactory implementation
sam2_option = False # whether to use SAM2 for segmentation
sam2_wo_bg = False # whether to use SAM2 without background
sam2_w_crop_option = False # whether to use SAM2 with cropping

# DexVLA specific settings
lang_intrs = "Insert the object into the target position"  
state_dim = 10  # 3D平移 + 6D旋转 + 1D夹爪宽度
action_dim = 10  # 3D平移 + 6D旋转 + 1D夹爪宽度

def get_target_id(eposide):
    # if eposide<=24:
    #     return np.array([0])
    # else:
    return np.array([0])

# def get_rebar_id(eposide_id):
#     if eposide_id <= 9 or 49>= eposide_id >= 22 or 104>= eposide_id >= 79 or 174>= eposide_id >= 154:
#         return "rebar1"
#     else:
#         return "rebar2"

os.makedirs(output_path, exist_ok=True)
# Search all subfolders starting with "run" and sort them
subfolders = sorted([folder for folder in os.listdir(input_path) if folder.startswith('run') and os.path.isdir(os.path.join(input_path, folder))])

# Initialize lists to store the merged data and episode end indices
all_ee_data = []
all_action_data = []
episode_ends = []
timestamps = []
current_index = 0

# To avoid singularity of axis angle representation
# Convert a position and axis-angle representation to a transformation matrix
def pos_axis2mat(pos_axis_angle):
    mat = np.eye(4)
    rot = Rotation.from_rotvec(pos_axis_angle[3:6])
    rot_mat = rot.as_matrix()
    mat[:3, :3] = rot_mat
    mat[:3, 3] = pos_axis_angle[:3]
    return  mat

# Convert a transformation matrix to a position and axis-angle representation
def mat2pos_axis(mat):
    pose_6D = np.zeros(6)
    mat = mat.reshape(4, 4, order='F')
    rot_mat = mat[:3, :3]
    trans_vec = mat[:3, 3]
    rot = Rotation.from_matrix(rot_mat)
    rot_vec = rot.as_rotvec()
    pose_6D[:3] = trans_vec
    pose_6D[3:] = rot_vec
    return pose_6D

# Convert a position and axis-angle representation to a transformation matrix
def avoid_singularity(Input, type = "mat"):
    A = np.array(
        [[1, 0, 0, 0],
         [0, -1, 0, 0.0],
         [0, 0, -1, 0.0],
         [0, 0, 0, 1.0]])
    if type == "mat":
        mat_new = A @ Input
        pos_angles_new = mat2pos_axis(mat_new)
        return pos_angles_new
    elif type == "pos_axis_angle":
        mat = pos_axis2mat(Input)
        mat_new = A @ mat  # A.T
        pos_angles_new = mat2pos_axis(mat_new)
        return pos_angles_new

    else:
        raise NotImplementedError(f"Type {type} is not implemented")

# repetitive paths can be caused by the incorrect data collection, so we can filter them
def filter_repetitives(rgb_paths): # if we incorrectly add repetitive paths, this can help
    filtered_rgb_paths = list(set(rgb_paths))
    filtered_rgb_paths.sort(key=rgb_paths.index)
    if len(rgb_paths) != len(filtered_rgb_paths): # delete competitive path
        print(f"Number of elements before filtering: {len(rgb_paths)}")
        print(f"Number of elements after filtering: {len(filtered_rgb_paths)}")
    return filtered_rgb_paths

# Convert absolute position and axis-angle representation to relative position and axis-angle representation
def abs2relative(ee_gripper_data, ref_pos, type="6D"):
    from scipy.spatial.transform import Rotation
    if type == "6D":
        # Take the first 6 values of the first row as the original pose
        initial_xyz = ref_pos[:3]
        initial_axis_angle = ref_pos[3:6]
        initial_quat = Rotation.from_rotvec(initial_axis_angle)
        # xyz
        relative_pose = ee_gripper_data.copy()
        relative_pose[:, :3] -= initial_xyz
        # rx, ry, rz
        for i in range(relative_pose.shape[0] - 1):
            abs_axis_angle = relative_pose[i, 3:6]
            abs_quat = Rotation.from_rotvec(abs_axis_angle)

            quat_diff = abs_quat * initial_quat.inv()
            relative_pose[i, 3:6] = quat_diff.as_rotvec()
        # The last column (gripper state) remains unchanged
    elif type == "pos":
        initial_xyz = ref_pos[:3]
        relative_pose = ee_gripper_data.copy()
        relative_pose[:, :3] -= initial_xyz
    else:
        raise NotImplementedError
    return relative_pose

# Convert absolute position and axis-angle representation to relative position and axis-angle representation
def abs2delta(ee_gripper_data, type="6D"):
    if type == "6D":
        # Take the first 6 values of the first row as the original pose
        initial_xyz = ee_gripper_data[0, :3]
        initial_axis_angle = ee_gripper_data[0, 3:6]
        initial_quat = Rotation.from_rotvec(initial_axis_angle)
        # xyz
        relative_pose = ee_gripper_data.copy()
        relative_pose[:, :3] -= initial_xyz
        # rx, ry, rz
        for i in range(relative_pose.shape[0] - 1):
            abs_axis_angle = relative_pose[i, 3:6]
            abs_quat = Rotation.from_rotvec(abs_axis_angle)

            quat_diff = abs_quat * initial_quat.inv()
            relative_pose[i, 3:6] = quat_diff.as_rotvec()
        # The last column (gripper state) remains unchanged
    # if type == "pos":
    #     initial_xyz = ee_gripper_data[0, :3]
    #     relative_pose = ee_gripper_data.copy()
    #     relative_pose[:, :3] -= initial_xyz
    else:
        raise NotImplementedError
    return relative_pose

# Convert 6D axis-angle to 6D Euler angles (for DexVLA single-arm requirement)
def axis_angle_to_euler_6d(axis_angle):
    """
    将轴角表示转换为6D欧拉角表示
    符合DexVLA单臂要求：使用Euler angles
    
    Args:
        axis_angle: [rx, ry, rz] in radians (轴角表示)
    Returns:
        euler_6d: [roll, pitch, yaw, roll2, pitch2, yaw2] in radians
        
    注意：DexVLA单臂使用Euler angles，不是6D rotation representation
    """
    # 转换为旋转矩阵
    rotation_matrix = Rotation.from_rotvec(axis_angle).as_matrix()
    
    # 转换为欧拉角 (ZYX convention)
    euler_angles = Rotation.from_matrix(rotation_matrix).as_euler('zyx', degrees=False)
    
    # 创建6D欧拉角表示 (duplicate for 6D)
    euler_6d = np.concatenate([euler_angles, euler_angles])
    
    return euler_6d

# Convert a position and axis-angle representation to a transformation matrix
def get_valid_length(arr):
    """returns the index of the first timestep that is all NaN (i.e., valid length)"""
    if arr.ndim == 1:
        mask = ~np.isnan(arr)
        return np.sum(mask)
    else:
        mask = ~np.isnan(arr).any(axis=tuple(range(1, arr.ndim)))
        return np.argmax(~mask) if np.any(~mask) else arr.shape[0]


# Loop through each subfolder and process data 
for episode_idx, folder in enumerate(subfolders):
    print(f"Processing {folder}...")
    # Load data 
    ee_states_path = os.path.join(input_path, folder, 'demo_ee_states.npz')
    gripper_path = os.path.join(input_path, folder, 'demo_gripper_states.npz')
    FT_raw_path = os.path.join(input_path, folder, 'demo_FT_raw.npz')
    FT_processed_path = os.path.join(input_path, folder, 'demo_FT_processed.npz')
    action_path = os.path.join(input_path, folder, 'demo_action.npz')
    action_grasp_path = os.path.join(input_path, folder, 'demo_action_grasp.npz')
    action_hot = os.path.join(input_path, folder, 'demo_action_hot.npz')
    # Load camera data
    camera_paths = []
    npz_list = os.listdir(os.path.join(input_path, folder))
    npz_list.sort()
    for idx, file in enumerate(npz_list):
        if "camera" in file and 'npz' in file:
            cam_idx = file.split("_")[2].split(".")[0]
            if cam_idx in cam_idxes:
                camera_paths.append(os.path.join(input_path, folder, f'demo_camera_{cam_idx}.npz'))
    # Load end effector and gripper data
    ee_data_raw = np.load(ee_states_path, allow_pickle=True)['data'] # Ufactory: [x,y,z,rx,ry,rz], rotation is in rad 
    # Convert translation from mm → m 
    ee_data_raw[:, :3] /= 1000.0
    # Load gripper data
    gripper_data = np.load(gripper_path, allow_pickle=True)['data']
    episode_len = ee_data_raw.shape[0] 

    # To avoid singularity of axis angle representation, make ee rot same as the robot base rot
    if img_only:
        ee_gripper_data = None
    else:
        if action_type == "abs":
            # 生成10维数据：3D平移 + 6D欧拉角 + 1D夹爪宽度 (符合DexVLA单臂要求)
            ee_data_10d = np.zeros((episode_len, 10))
            
            for i in range(episode_len):
                # 3D平移 (x, y, z)
                ee_data_10d[i, :3] = ee_data_raw[i, :3] / 1000.0  # mm → m
                
                # 6D欧拉角 (roll, pitch, yaw, roll2, pitch2, yaw2) - DexVLA单臂使用Euler angles
                axis_angle = ee_data_raw[i, 3:6]  # [rx, ry, rz]
                euler_6d = axis_angle_to_euler_6d(axis_angle)
                ee_data_10d[i, 3:9] = euler_6d
                
                # 1D夹爪宽度
                ee_data_10d[i, 9] = gripper_data[i]
            
            ee_gripper_data = ee_data_10d  # 现在shape是 (T, 10)
            
            print(f"Generated 10D data shape: {ee_gripper_data.shape}")
            print(f"Sample 10D data: {ee_gripper_data[0]}")
            print(f"Translation (m): {ee_gripper_data[0, :3]}")
            print(f"6D Euler angles (rad): {ee_gripper_data[0, 3:9]}")
            print(f"Gripper width: {ee_gripper_data[0, 9]}")

        else:
            raise NotImplementedError

    # ----------------------- action states ----------------------- #
    # If we use relative action, we need to convert the absolute action to relative action
    if action_type == "abs":
        # 直接加载demo_action.npz，然后转换为10维数据
        action_data_raw = np.load(action_path, allow_pickle=True)['data']  # shape: (T, 7)
        print(f"Loaded action data shape: {action_data_raw.shape}")
        
        # 转换为10维数据：3D平移 + 6D欧拉角 + 1D夹爪宽度 (符合DexVLA单臂要求)
        action_data_10d = np.zeros((episode_len, 10))
        
        for i in range(episode_len):
            # 3D平移 (x, y, z) - 从mm转换为m
            action_data_10d[i, :3] = action_data_raw[i, :3] / 1000.0
            
            # 6D欧拉角 (roll, pitch, yaw, roll2, pitch2, yaw2) - DexVLA单臂使用Euler angles
            axis_angle = action_data_raw[i, 3:6]  # [rx, ry, rz]
            euler_6d = axis_angle_to_euler_6d(axis_angle)
            action_data_10d[i, 3:9] = euler_6d
            
            # 1D夹爪宽度
            action_data_10d[i, 9] = action_data_raw[i, 6]
        
        action_data = action_data_10d  # 现在shape是 (T, 10)
        
        print(f"Converted action data shape: {action_data.shape}")
        print(f"Action translation sample (m): {action_data[0, :3]}")
        print(f"Action 6D Euler angles sample (rad): {action_data[0, 3:9]}")
        print(f"Action gripper sample: {action_data[0, 9]}")
        print(f"Qpos data shape: {ee_gripper_data.shape}")
        print(f"Qpos gripper column sample: {ee_gripper_data[:, -1][:5]}")
        
        # 验证数据一致性
        print(f"\nData validation:")
        print(f"  Action gripper values: {np.unique(action_data[:, -1])}")
        print(f"  Qpos gripper values: {np.unique(ee_gripper_data[:, -1])}")
        print(f"  Action translation range (m): {np.min(action_data[:, :3]):.3f} to {np.max(action_data[:, :3]):.3f}")
        print(f"  Qpos translation range (m): {np.min(ee_gripper_data[:, :3]):.3f} to {np.max(ee_gripper_data[:, :3]):.3f}")
    else:
        raise NotImplementedError

    # ----------------------- force and torque ----------------------- #
    FT_raw_data = np.load(FT_raw_path, allow_pickle=True)['data']
    FT_processed_data = np.load(FT_processed_path, allow_pickle=True)['data']

    # ----------------------- images ----------------------- #
    assert len(camera_paths) == len(camera_names)
    all_image_data = {}

    for idx, cam_name in enumerate(camera_names):
        if sam2_option:
            pass
            # segmentor = Sam2()
            # segmentor.initialize(slot_id=f"{get_rebar_id(episode_idx)}_{cam_name}_slot5", single_class=False, manual=False)
        camera_data = np.load(camera_paths[idx], allow_pickle=True)
        rgb_paths = [os.path.join(input_path,"images", item['color_img_name'].split("/")[-2], item['color_img_name'].split("/")[-1] + ".jpg") for item in camera_data["data"]]
        
        # if we use sam2 to segment the image, we need to crop the image
        if sam2_option:
            rgb_data_crop_1 = []
            rgb_data_crop_2 = []
            rgb_data_crop_3 = []
        else:
            rgb_data_single_cam = []

        # rgb_paths = filter_repetitives(rgb_paths) # just for debugging

        # Check if the number of images matches the episode length
        if crop_wo_sam2_option:
            crop_region = crop_list[0]
            for i, image_path in enumerate(rgb_paths):
                img = cv2.imread(image_path)
                x1, y1, x2, y2 = crop_region
                img = img[y1:y2, x1:x2]
                # Ensure the cropped image matches the desired crop size
                if img.shape[:2] != (y2 - y1, x2 - x1):
                    raise ValueError(f"Unexpected crop size: {img.shape[:2]}, expected ({image_h}, {image_w})")
                rgb_data_single_cam.append(img)

        elif sam2_option:
            pass
            # for i, image_path in enumerate(rgb_paths):
            #     img = cv2.imread(image_path) # read as BGR
            #     if sam2_w_crop_option: # remove useless regions
            #         crop_region = crop_list[0]
            #         x1, y1, x2, y2 = crop_region
            #         img = img[y1:y2, x1:x2]
            #     img = img.transpose(2, 0, 1)  # Convert (H, W, C) -> (C, H, W)
            #     crop_images = segmentor.get_sub_images(img, slot_id=f"{get_rebar_id(episode_idx)}_{cam_name}_slot5",
            #                                                 wo_bg=sam2_wo_bg,
            #                                                bbox_height=320, bbox_width=480,
            #                                                h_shift=0,  w_shift=0, data_aug = True)   # list, img: (H, W, C)
            #     rgb_data_crop_1.append(crop_images[0])
            #     rgb_data_crop_2.append(crop_images[0])
            #     rgb_data_crop_3.append(crop_images[0])

        else:
            for i, image_path in enumerate(rgb_paths):
                img = cv2.imread(image_path)
                if img.shape[:2] != (image_h, image_w):
                    # Resize the image if dimensions do not match
                    img = cv2.resize(img, (image_w, image_h), interpolation=cv2.INTER_LINEAR)
                rgb_data_single_cam.append(img) # img: (H, W, C)

        if sam2_option:
            pass

            # rgb_data_crop_1 = np.array(rgb_data_crop_1)
            # rgb_data_crop_2 = np.array(rgb_data_crop_2)
            # rgb_data_crop_3 = np.array(rgb_data_crop_3)
            # all_image_data[f"{cam_name}_rgb_crop1"] = rgb_data_crop_1
            # all_image_data[f"{cam_name}_rgb_crop1"] = rgb_data_crop_1
            # all_image_data[f"{cam_name}_rgb_crop1"] = rgb_data_crop_1
        else:
            rgb_data_single_cam = np.array(rgb_data_single_cam)
            all_image_data[f"{cam_name}_rgb"] = rgb_data_single_cam

        #  We didn't implement the depth map
        # if camera_data["data"][0]["depth_img_name"] != "":
        #     depth_paths = [os.path.join(input_path, images, item['depth_img_name'].split("/")[-2],
        #                               item['depth_img_name'].split("/")[-1] + ".png") for item in camera_data["data"]]
        #     depth_data_single_cam = []
        #     # depth_paths = filter_repetitives(depth_paths)  # just for debugging
        #     for i, image_path in enumerate(rgb_paths):
        #         img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #         depth_data_single_cam.append(img)
        #         depth_data_single_cam.append(img)
        #         depth_data_single_cam.append(img)
        #     depth_data_single_cam = np.array(depth_data_single_cam)
        #     all_image_data[f"{cam_name}_depth"] = depth_data_single_cam

    # ----------------------- target ID ----------------------- #
    target = get_target_id(episode_idx)

    # create hdf5
    if img_only:
        data_dict = {
            'action': [action_data],
            "target": [target]
        }
    else:
        # 注意：数据字典将在循环内更新，这里只是初始化
        data_dict = {
            'action':  [action_data],
            "target": [target],
            "observations/FT_raw": [FT_raw_data],
            "observations/FT_processed": [FT_processed_data],
            # "action_hot": [action_hot_data],
            'observations/qpos':  [ee_gripper_data],
            'observations/qvel': [np.zeros_like(ee_gripper_data)],  # add qvel
            'is_edited': [np.array([False])]  # add is_edited flag
        }

    for idx, cam_name in enumerate(camera_names):
        if sam2_option:
            data_dict[f'/observations/images/{cam_name}_crop1'] = []
            data_dict[f'/observations/images/{cam_name}_crop1'].append(all_image_data[f"{cam_name}_rgb_crop1"])
            data_dict[f'/observations/images/{cam_name}_crop1'] = []
            data_dict[f'/observations/images/{cam_name}_crop1'].append(all_image_data[f"{cam_name}_rgb_crop1"])
            data_dict[f'/observations/images/{cam_name}_crop1'] = []
            data_dict[f'/observations/images/{cam_name}_crop1'].append(all_image_data[f"{cam_name}_rgb_crop1"])
        else:
            data_dict[f'/observations/images/{cam_name}'] = []
            data_dict[f'/observations/images/{cam_name}'].append(all_image_data[f"{cam_name}_rgb"])
            if "rs" in cam_name:
                data_dict[f'/observations/images/{cam_name}_depth'] = []
                data_dict[f'/observations/images/{cam_name}_depth'].append(all_image_data[f"{cam_name}_depth"])

        arrays_to_trim = [
            action_data,
            ee_gripper_data,
            FT_raw_data,
            FT_processed_data,
        ]
        for cam_name in camera_names:
            arrays_to_trim.append(all_image_data[f"{cam_name}_rgb"])

        valid_lens = [get_valid_length(arr) for arr in arrays_to_trim]
        valid_len = min(valid_lens)
        if valid_len == 0:
            print(f"{folder} no valid data, skipping...")
            continue

        # trim the arrays to the valid length
        action_data = action_data[:valid_len]
        ee_gripper_data = ee_gripper_data[:valid_len]
        FT_raw_data = FT_raw_data[:valid_len]
        FT_processed_data = FT_processed_data[:valid_len]
        for cam_name in camera_names:
            all_image_data[f"{cam_name}_rgb"] = all_image_data[f"{cam_name}_rgb"][:valid_len]

        # 确保action和qpos长度匹配，并生成正确的格式
        # 注意：action和qpos是独立的数据源，应该有相同的长度
        # action[t] = demo_action.npz中的第t帧数据
        # qpos[t] = demo_ee_states.npz中的第t帧数据
        
        # 直接使用裁剪后的数据，不需要切片
        action_data_final = action_data      # 直接使用action数据
        qpos_data_final = ee_gripper_data    # 直接使用qpos数据
        
        # 验证长度是否一致
        if action_data_final.shape[0] != qpos_data_final.shape[0]:
            print(f"Warning: Action and qpos have different lengths!")
            print(f"  Action length: {action_data_final.shape[0]}")
            print(f"  Qpos length: {qpos_data_final.shape[0]}")
            # 取较短的长度作为最终长度
            final_valid_len = min(action_data_final.shape[0], qpos_data_final.shape[0])
            action_data_final = action_data_final[:final_valid_len]
            qpos_data_final = qpos_data_final[:final_valid_len]
        else:
            final_valid_len = action_data_final.shape[0]
        
        print(f"Final action shape: {action_data_final.shape}")
        print(f"Final qpos shape: {qpos_data_final.shape}")
        print(f"Final action gripper sample: {action_data_final[:, -1][:5]}")
        print(f"Final qpos gripper sample: {qpos_data_final[:, -1][:5]}")
        print(f"Final valid length: {final_valid_len}")
        
        # 裁剪所有数据到最终长度
        action_data_final = action_data_final[:final_valid_len]
        qpos_data_final = qpos_data_final[:final_valid_len]
        FT_raw_data = FT_raw_data[:final_valid_len]
        FT_processed_data = FT_processed_data[:final_valid_len]
        
        # 裁剪图像数据到正确长度
        for cam_name in camera_names:
            all_image_data[f"{cam_name}_rgb"] = all_image_data[f"{cam_name}_rgb"][:final_valid_len]
            print(f"Image {cam_name} trimmed to: {all_image_data[f'{cam_name}_rgb'].shape}")
        
        print(f"All data trimmed to final length: {final_valid_len}")
        print(f"  FT_raw_data: {FT_raw_data.shape}")
        print(f"  FT_processed_data: {FT_processed_data.shape}")
        
        # 生成qvel数据并裁剪到正确长度
        qvel_data = np.zeros_like(qpos_data_final)  # 使用qpos的形状
        print(f"Qvel data shape: {qvel_data.shape}")
        
        # 更新数据字典
        data_dict['action'] = [action_data_final]
        data_dict['observations/qpos'] = [qpos_data_final]
        data_dict['observations/qvel'] = [qvel_data]  # 使用裁剪后的qvel
        data_dict['observations/FT_raw'] = [FT_raw_data]
        data_dict['observations/FT_processed'] = [FT_processed_data]

        # 验证数据一致性
        assert action_data_final.shape[0] == qpos_data_final.shape[0] == final_valid_len, f"Shape is incorrect: action={action_data_final.shape[0]}, qpos={qpos_data_final.shape[0]}, final_valid_len={final_valid_len}"
        assert FT_raw_data.shape[0] == FT_processed_data.shape[0] == final_valid_len, f"FT data length mismatch: FT_raw={FT_raw_data.shape[0]}, FT_processed={FT_processed_data.shape[0]}, final_valid_len={final_valid_len}"
        for idx in range(len(list(all_image_data.values()))):
            assert list(all_image_data.values())[idx].shape[0] == final_valid_len, f"Image {idx} length {list(all_image_data.values())[idx].shape[0]} != final_valid_len {final_valid_len}"

        assert max_timesteps >= final_valid_len, f"Episode length {final_valid_len} is beyond max_timesteps {max_timesteps}"

        # 创建HDF5文件
        with h5py.File(os.path.join(output_path,f'episode_{episode_idx + hdf5_start_idx}.hdf5'), 'w') as root: # {episode_idx}
            root.attrs['sim'] = True 
            obs = root.create_group('observations')
            image = obs.create_group('images')

            for cam_name in camera_names:
                if sam2_option:
                    _ = image.create_dataset(
                        f'{cam_name}_crop1',
                        (final_valid_len, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                        dtype='uint8',  # Image data type (uint8 for images)
                        chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                        fillvalue=np.nan
                    )
                    _ = image.create_dataset(
                        f'{cam_name}_crop1',
                        (final_valid_len, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                        dtype='uint8',  # Image data type (uint8 for images)
                        chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                        fillvalue=np.nan
                    )
                    _ = image.create_dataset(
                        f'{cam_name}_crop1',
                        (final_valid_len, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                        dtype='uint8',  # Image data type (uint8 for images)
                        chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                        fillvalue=np.nan
                    )
                else:
                    _ = image.create_dataset(
                        cam_name,
                        (final_valid_len, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                        dtype='uint8',  # Image data type (uint8 for images)
                        chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                        fillvalue = np.nan
                    )
                if "rs" in cam_name:
                    _ = image.create_dataset(
                        f"{cam_name}_depth",
                        (final_valid_len, image_h, image_w),  # Shape: (timesteps, height, width, channels)
                        dtype='uint16',  # Image data type (uint8 for images)
                        chunks=(1, image_h, image_w),  # Chunk size for better performance
                        fillvalue = np.nan
                    )
            
            # Fill the image datasets with data
            qpos = obs.create_dataset('qpos', (final_valid_len, 10), fillvalue=np.nan) # 10-dimensional: 3D translation + 6D Euler angles + 1D gripper (DexVLA single-arm)
            qvel = obs.create_dataset('qvel', (final_valid_len, 10), fillvalue=np.nan)
            FT_raw = obs.create_dataset('FT_raw', (final_valid_len, 6), fillvalue=np.nan)
            FT_processed = obs.create_dataset('FT_processed', (final_valid_len, 6), fillvalue=np.nan)
            action = root.create_dataset('action', (final_valid_len, 10), fillvalue=np.nan) # 10-dimensional: 3D translation + 6D Euler angles + 1D gripper (DexVLA single-arm)
            action_hot = root.create_dataset('action_hot', (final_valid_len, 1), fillvalue=np.nan)
            target_hot = root.create_dataset('target', (1), fillvalue=np.nan)
            is_edited = root.create_dataset('is_edited', (1), fillvalue=False)  # add is_edited flag
            
            # add language features
            root.create_dataset("language_raw", data=[lang_intrs])
            # if we use BERT features, we can add dummy features here
            dummy_bert_features = np.zeros((1, 768))  # assuming BERT features are 768-dimensional
            root.create_dataset("distill_bert_lang", data=dummy_bert_features)

            # Fill the datasets with data - 添加缺失的sub_reason_distilbert和substep_reasonings功能
            substep_reasoning_str = "Insert the rebar into the target position"
            substep_reasonings = [substep_reasoning_str] * final_valid_len
            sub_reason_distilbert = []
            with torch.no_grad():
                for r in substep_reasonings:
                    inputs = tokenizer(r, return_tensors="pt")
                    outputs = model(**inputs)
                    emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # (768,)
                    sub_reason_distilbert.append(emb.astype(np.float16))  # float16

            sub_reason_distilbert = np.array(sub_reason_distilbert)  # (final_valid_len, 768)
            sub_reason_distilbert = sub_reason_distilbert[:, None, :]  # (final_valid_len, 1, 768)

            root.create_dataset("substep_reasonings", data=np.array(substep_reasonings, dtype=object))
            root.create_dataset("sub_reason_distilbert", data=sub_reason_distilbert)

            for name, array in data_dict.items():
                dataset_path = name
                if name.startswith('/'):
                    dataset_path = name[1:] 
                
                if dataset_path in root:
                    try:
                        root[dataset_path][:array[0].shape[0], ...] = array[0]
                        print(f"Successfully wrote {name} with shape {array[0].shape}")
                    except Exception as e:
                        print(f"Error writing {name}: {e}")
                        raise Exception
                else:
                    print(f"{dataset_path} not exist in root")
                    raise NotImplementedError

print("All data has been processed, merged, and saved.")