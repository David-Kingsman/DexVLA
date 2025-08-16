import numpy as np
import os
from scipy.spatial.transform import Rotation
import h5py
import cv2
# import transform_utils
import copy
import sys
# sys.path.append("/mnt/sda1/act") 
# make the current NumPy’s core visible under the old name
from transformers import BertTokenizer, BertModel
import torch

# Import ufactory SDK for inverse kinematics
try:
    from xarm.wrapper import XArmAPI
    UFACTORY_AVAILABLE = True
    print("ufactory SDK imported successfully")
except ImportError:
    print("Warning: ufactory SDK not available. Please install xarm-python-sdk")
    UFACTORY_AVAILABLE = False

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

"""
Convert the demos from Deoxys to hdf5 format used by ACT
This script converts 6D pose data to joint angles using ufactory SDK inverse kinematics:

Input: 6D pose data (x, y, z, rx, ry, rz) + gripper from demo_ee_states.npz
Process: Convert 6D pose to joint angles using xarm6 inverse kinematics
Output: action[t] = target joint angles for next timestep, qpos[t] = current joint angles
Compatible with DexVLA absolute action training
"""
# ----------------------- !!! Setting !!! -------------------- #
input_path = 'data/rebar_insertion' 
output_path = "data/rebar_insertion_hdf5" 
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

# ufactory arm settings for inverse kinematics
use_real_arm_for_ik = False  # Set to True if you want to use real arm for IK calculations
arm_ip = "192.168.1.100"     # Replace with your arm's IP address

# DexVLA specific settings - Using joint control for xarm6
lang_intrs = "Insert the object into the target position"  
state_dim = 7  # 6 joint angles + 1 gripper
action_dim = 7  # 6 joint angles + 1 gripper

"""
Usage Instructions:

1. Data Format Requirements:
   - demo_ee_states.npz: 6D pose data [x, y, z, rx, ry, rz] (units: m, rad)
   - demo_gripper_states.npz: gripper state data
   - demo_camera_*.npz: camera data

2. Inverse Kinematics Settings:
   - use_real_arm_for_ik = False: use mock IK (recommended for testing)
   - use_real_arm_for_ik = True: use real arm for IK calculations
   - arm_ip: set your xarm6 IP address

3. Output Format:
   - action[t]: target joint angles for timestep t+1 (7D: 6 joints + 1 gripper)
   - qpos[t]: current joint angles at timestep t (7D: 6 joints + 1 gripper)

4. Important Notes:
   - Ensure 6D pose data is within xarm6 workspace
   - If using real arm, ensure network connection is stable
   - Mock IK provides reasonable joint angles but may not be precise
"""

def create_mock_arm_instance():
    """
    Create a mock arm instance for testing inverse kinematics
    This is useful when you don't have access to a real arm or want to test offline
    """
    class MockArm:
        def __init__(self):
            # Mock joint limits and workspace for xarm6
            self.joint_limits = [
                [-360, 360],  # Joint 1 limits in degrees
                [-118, 120],  # Joint 2 limits in degrees  
                [-225, 11],   # Joint 3 limits in degrees
                [-360, 360],  # Joint 4 limits in degrees
                [-97, 180],   # Joint 5 limits in degrees
                [-360, 360]   # Joint 6 limits in degrees
            ]
        
        def get_inverse_kinematics(self, pose, input_is_radian=True, return_is_radian=True):
            """
            Mock inverse kinematics - returns reasonable joint angles
            In practice, you should use the real arm or a proper IK solver
            
            Args:
                pose: 4x4 transformation matrix flattened to list
                input_is_radian: ignored in mock
                return_is_radian: ignored in mock
            
            Returns:
                List of 6 joint angles in radians
            """
            # Convert pose matrix back to numpy array
            pose_matrix = np.array(pose).reshape(4, 4)
            
            # Extract position and rotation
            position = pose_matrix[:3, 3]
            rotation_matrix = pose_matrix[:3, :3]
            
            # Simple heuristic for mock IK:
            # - Joint 1: based on x, y position (yaw)
            # - Joint 2, 3: based on z position and reach
            # - Joint 4, 5, 6: based on rotation matrix
            
            # Calculate joint 1 (yaw) from x, y position
            joint1 = np.arctan2(position[1], position[0])
            
            # Calculate joint 2, 3 based on z and reach
            xy_distance = np.sqrt(position[0]**2 + position[1]**2)
            z_offset = position[2] - 0.5  # Assuming base height is 0.5m
            
            # Simple inverse kinematics for 2-link arm
            L1, L2 = 0.4, 0.4  # Mock link lengths
            cos_theta3 = (xy_distance**2 + z_offset**2 - L1**2 - L2**2) / (2 * L1 * L2)
            cos_theta3 = np.clip(cos_theta3, -1, 1)
            joint3 = np.arccos(cos_theta3)
            
            # Calculate joint 2
            joint2 = np.arctan2(z_offset, xy_distance) - np.arctan2(L2 * np.sin(joint3), L1 + L2 * np.cos(joint3))
            
            # Calculate joint 4, 5, 6 from rotation matrix
            # This is a simplified approach - in practice you'd use proper IK
            joint4 = 0.0
            joint5 = 0.0
            joint6 = 0.0
            
            # Ensure joints are within limits
            joints = [joint1, joint2, joint3, joint4, joint5, joint6]
            for i, (joint, limits) in enumerate(zip(joints, self.joint_limits)):
                joint_deg = np.degrees(joint)
                if joint_deg < limits[0]:
                    joints[i] = np.radians(limits[0])
                elif joint_deg > limits[1]:
                    joints[i] = np.radians(limits[1])
            
            return joints
        
        def disconnect(self):
            pass  # Mock disconnect
    
    return MockArm()

def convert_6d_pose_to_joint_angles(pose_6d, arm_instance):
    """
    Convert 6D pose (x, y, z, rx, ry, rz) to joint angles using ufactory SDK
    
    Args:
        pose_6d: numpy array of shape (6,) containing [x, y, z, rx, ry, rz]
        arm_instance: XArmAPI instance for inverse kinematics
    
    Returns:
        joint_angles: numpy array of shape (6,) containing joint angles in radians
    """
    if not UFACTORY_AVAILABLE:
        raise RuntimeError("ufactory SDK not available")
    
    try:
        # Extract position and rotation
        x, y, z, rx, ry, rz = pose_6d
        
        # Convert axis-angle to rotation matrix
        rotation_matrix = Rotation.from_rotvec([rx, ry, rz]).as_matrix()
        
        # Create pose matrix
        pose_matrix = np.eye(4)
        pose_matrix[:3, :3] = rotation_matrix
        pose_matrix[:3, 3] = [x, y, z]
        
        # Use ufactory SDK inverse kinematics
        joint_angles = arm_instance.get_inverse_kinematics(
            pose=pose_matrix.flatten().tolist(),
            input_is_radian=True,
            return_is_radian=True
        )
        
        if joint_angles is None or len(joint_angles) < 6:
            raise ValueError("Inverse kinematics failed to return valid joint angles")
        
        return np.array(joint_angles[:6])  # Return first 6 joint angles
        
    except Exception as e:
        print(f"Error in inverse kinematics: {e}")
        # Return a default pose if IK fails
        return np.zeros(6)

def batch_convert_6d_pose_to_joint_angles(poses_6d, arm_instance):
    """
    Convert a batch of 6D poses to joint angles
    
    Args:
        poses_6d: numpy array of shape (T, 6) containing 6D poses
        arm_instance: XArmAPI instance for inverse kinematics
    
    Returns:
        joint_angles: numpy array of shape (T, 6) containing joint angles
    """
    joint_angles = np.zeros((poses_6d.shape[0], 6))
    
    for i, pose in enumerate(poses_6d):
        try:
            joint_angles[i] = convert_6d_pose_to_joint_angles(pose, arm_instance)
            if i % 10 == 0:  # Progress indicator
                print(f"Converted {i+1}/{poses_6d.shape[0]} poses to joint angles")
        except Exception as e:
            print(f"Failed to convert pose {i}: {e}")
            # Use previous valid joint angles if available
            if i > 0:
                joint_angles[i] = joint_angles[i-1]
            else:
                joint_angles[i] = np.zeros(6)
    
    return joint_angles

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
    
    # ----------------------- 6D Pose to Joint Angles Conversion ----------------------- #
    # Load 6D pose data (end-effector positions and orientations)
    ee_states_path = os.path.join(input_path, folder, 'demo_ee_states.npz')
    gripper_path = os.path.join(input_path, folder, 'demo_gripper_states.npz')
    
    # Load 6D pose data
    ee_data_raw = np.load(ee_states_path, allow_pickle=True)['data']  # shape: (T, 6) - [x, y, z, rx, ry, rz]
    gripper_data = np.load(gripper_path, allow_pickle=True)['data']   # shape: (T, 1)
    
    print(f"Loaded 6D pose data: {ee_data_raw.shape}")
    print(f"Loaded gripper data: {gripper_data.shape}")
    
    # Convert translation from mm to m if needed
    if np.max(np.abs(ee_data_raw[:, :3])) > 10:  # If positions are in mm (likely > 10m)
        ee_data_raw[:, :3] /= 1000.0
        print("Converted positions from mm to m")
    
    # Initialize ufactory arm instance for inverse kinematics
    if UFACTORY_AVAILABLE:
        if use_real_arm_for_ik:
            # Use real arm for IK calculations
            arm_instance = XArmAPI(arm_ip)
            try:
                arm_instance.connect()
                print(f"Connected to ufactory arm at {arm_ip} for IK calculations")
            except Exception as e:
                print(f"Warning: Could not connect to real arm at {arm_ip}: {e}")
                print("Falling back to mock instance for IK")
                arm_instance = create_mock_arm_instance()
        else:
            # Use mock instance for testing (faster and safer)
            arm_instance = create_mock_arm_instance()
            print("Using mock arm instance for IK calculations (set use_real_arm_for_ik=True to use real arm)")
    else:
        raise RuntimeError("ufactory SDK not available for inverse kinematics")
    
    # Convert 6D poses to joint angles using inverse kinematics
    print(f"Converting {ee_data_raw.shape[0]} 6D poses to joint angles...")
    joint_angles_6d = batch_convert_6d_pose_to_joint_angles(ee_data_raw, arm_instance)
    
    # Combine joint angles with gripper data
    joint_data = np.concatenate((joint_angles_6d, gripper_data.reshape(-1, 1)), axis=1)  # shape: (T, 7)
    
    # Convert to action format: action[t] = qpos[t+1] (target joint angles for next timestep)
    action_data = joint_data[1:]           # action[t] = target joint angles for t+1
    ee_gripper_data = joint_data[:-1]      # qpos[t] = current joint angles at t
    
    # Handle edge case: ensure action_data and ee_gripper_data have same length
    if action_data.shape[0] < ee_gripper_data.shape[0]:
        action_data = np.vstack([action_data, joint_data[-1]])
    
    # Verify data dimensions for joint control
    print(f"6D pose data shape: {ee_data_raw.shape}")
    print(f"Joint angles shape: {joint_angles_6d.shape}")
    print(f"Action data shape: {action_data.shape}")
    print(f"Qpos data shape: {ee_gripper_data.shape if ee_gripper_data is not None else 'None'}")
    assert action_data.shape[1] == 7, f"Action data should have 7 dimensions (6 joints + 1 gripper), got {action_data.shape[1]}"
    if ee_gripper_data is not None:
        assert ee_gripper_data.shape[1] == 7, f"Qpos data should have 7 dimensions (6 joints + 1 gripper), got {ee_gripper_data.shape[1]}"
    
    # Clean up arm connection
    if UFACTORY_AVAILABLE and hasattr(arm_instance, 'disconnect'):
        try:
            arm_instance.disconnect()
        except:
            pass
    
    # ----------------------- force and torque ----------------------- #
    FT_raw_path = os.path.join(input_path, folder, 'demo_FT_raw.npz')
    FT_processed_path = os.path.join(input_path, folder, 'demo_FT_processed.npz')
    
    # Load camera data
    camera_paths = []
    npz_list = os.listdir(os.path.join(input_path, folder))
    npz_list.sort()
    for idx, file in enumerate(npz_list):
        if "camera" in file and 'npz' in file:
            cam_idx = file.split("_")[2].split(".")[0]
            if cam_idx in cam_idxes:
                camera_paths.append(os.path.join(input_path, folder, f'demo_camera_{cam_idx}.npz'))

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
            #     rgb_data_crop_2.append(crop_images[1])
            #     rgb_data_crop_3.append(crop_images[2])

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
            # all_image_data[f"{cam_name}_rgb_crop2"] = rgb_data_crop_2
            # all_image_data[f"{cam_name}_rgb_crop3"] = rgb_data_crop_3
        else:
            rgb_data_single_cam = np.array(rgb_data_single_cam)
            all_image_data[f"{cam_name}_rgb"] = rgb_data_single_cam

        #  We didn't implement the depth map
        # if camera_data["data"][0]["depth_img_name"] != "":
        #     depth_paths = [os.path.join(input_path, images, item['depth_img_name'].split("/")[-2],
        #                               item['depth_img_name'].split("/")[-1] + ".png") for item in camera_data["data"]]
        #     depth_data_single_cam = []
        #     # depth_paths = filter_repetitives(depth_paths)  # just for debugging
        #     for i, image_path in enumerate(depth_paths):
        #         img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
        #         depth_data_single_cam.append(img)
        #     depth_data_single_cam = np.array(depth_data_single_cam)
        #     all_image_data[f"{cam_name}_depth"] = depth_data_single_cam

    # ----------------------- target ID ----------------------- #
    target = get_target_id(episode_idx)

    # create hdf5
    # if img_only:
    #     data_dict = {
    #         'action': [action_data],
    #         "target": [target]
    #     }
    # else:
    #     data_dict = {
    #         'action':  [action_data],
    #         "target": [target],
    #         "observations/FT_raw": [FT_raw_data],
    #         "observations/FT_processed": [FT_processed_data],
    #         # "action_hot": [action_hot_data],
    #         'observations/qpos':  [ee_gripper_data],
    #         'observations/qvel': [np.zeros_like(ee_gripper_data)],  # add qvel
    #         'is_edited': [np.array([False])]  # add is_edited flag
    #     }
    data_dict = {}
    
    for idx, cam_name in enumerate(camera_names):
        if sam2_option:
            data_dict[f'/observations/images/{cam_name}_crop1'] = []
            data_dict[f'/observations/images/{cam_name}_crop1'].append(all_image_data[f"{cam_name}_rgb_crop1"])
            data_dict[f'/observations/images/{cam_name}_crop2'] = []
            data_dict[f'/observations/images/{cam_name}_crop2'].append(all_image_data[f"{cam_name}_rgb_crop2"])
            data_dict[f'/observations/images/{cam_name}_crop3'] = []
            data_dict[f'/observations/images/{cam_name}_crop3'].append(all_image_data[f"{cam_name}_rgb_crop3"])
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
       
        if img_only:
            data_dict = {
                'action': [action_data],
                "target": [target]
            }
        else:
            data_dict = {
                'action':  [action_data],
                "target": [target],
                "observations/FT_raw": [FT_raw_data],
                "observations/FT_processed": [FT_processed_data],
                # "action_hot": [action_hot_data[:valid_len]], 
                'observations/qpos':  [ee_gripper_data],
                'observations/qvel': [np.zeros_like(ee_gripper_data)],
                'is_edited': [np.array([False])]
            }

    assert action_data.shape[0] == ee_gripper_data.shape[0] == list(all_image_data.values())[0].shape[0], "Shape is incorrect"
    for idx in range(len(list(all_image_data.values()))):
        assert list(all_image_data.values())[0].shape[0] == list(all_image_data.values())[idx].shape[0], "Image numbers should be same"

    assert max_timesteps >= action_data.shape[0], f"Eposide length {action_data.shape[0]} is beyond max_timesteps {max_timesteps}"

    with h5py.File(os.path.join(output_path,f'episode_{episode_idx + hdf5_start_idx}.hdf5'), 'w') as root: # {episode_idx}
        root.attrs['sim'] = True 
        obs = root.create_group('observations')
        image = obs.create_group('images')

        for cam_name in camera_names:
            if sam2_option:
                _ = image.create_dataset(
                    f'{cam_name}_crop1',
                    (valid_len, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                    dtype='uint8',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                    fillvalue=np.nan
                )
                _ = image.create_dataset(
                    f'{cam_name}_crop2',
                    (valid_len, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                    dtype='uint8',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                    fillvalue=np.nan
                )
                _ = image.create_dataset(
                    f'{cam_name}_crop3',
                    (valid_len, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                    dtype='uint8',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                    fillvalue=np.nan
                )
            else:
                _ = image.create_dataset(
                    cam_name,
                    (valid_len, image_h, image_w, 3),  # Shape: (timesteps, height, width, channels)
                    dtype='uint8',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w, 3),  # Chunk size for better performance
                    fillvalue = np.nan
                )
            if "rs" in cam_name:
                _ = image.create_dataset(
                    f"{cam_name}_depth",
                    (valid_len, image_h, image_w),  # Shape: (timesteps, height, width, channels)
                    dtype='uint16',  # Image data type (uint8 for images)
                    chunks=(1, image_h, image_w),  # Chunk size for better performance
                    fillvalue = np.nan
                )
        
        # Fill the image datasets with data
        qpos = obs.create_dataset('qpos', (valid_len, 7), fillvalue=np.nan) # 7-dimentional ee pose and gripper state
        qvel = obs.create_dataset('qvel', (valid_len, 7), fillvalue=np.nan)
        FT_raw = obs.create_dataset('FT_raw', (valid_len, 6), fillvalue=np.nan)
        FT_processed = obs.create_dataset('FT_processed', (valid_len, 6), fillvalue=np.nan)
        action = root.create_dataset('action', (valid_len, 7), fillvalue=np.nan)
        action_hot = root.create_dataset('action_hot', (valid_len, 1), fillvalue=np.nan)
        target_hot = root.create_dataset('target', (1), fillvalue=np.nan)
        is_edited = root.create_dataset('is_edited', (1), fillvalue=False)  # add is_edited flag
        
        # add language features
        root.create_dataset("language_raw", data=[lang_intrs])
        # if we use BERT features, we can add dummy features here
        dummy_bert_features = np.zeros((1, 768))  # assuming BERT features are 768-dimensional
        root.create_dataset("distill_bert_lang", data=dummy_bert_features)

        # Fill the datasets with data
        substep_reasoning_str = "Insert the rebar into the target position"
        substep_reasonings = [substep_reasoning_str] * valid_len
        sub_reason_distilbert = []
        with torch.no_grad():
            for r in substep_reasonings:
                inputs = tokenizer(r, return_tensors="pt")
                outputs = model(**inputs)
                emb = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()  # (768,)
                sub_reason_distilbert.append(emb.astype(np.float16))  # float16

        sub_reason_distilbert = np.array(sub_reason_distilbert)  # (valid_len, 768)
        sub_reason_distilbert = sub_reason_distilbert[:, None, :]  # (valid_len, 1, 768)


        root.create_dataset("substep_reasonings", data=np.array(substep_reasonings, dtype=object))
        root.create_dataset("sub_reason_distilbert", data=sub_reason_distilbert) 

        for name, array in data_dict.items():
            dataset_path = name
            if name.startswith('/'):
                dataset_path = name[1:] 
            
            if dataset_path in root:
                try:
                    root[dataset_path][:array[0].shape[0], ...] = array[0]
                except Exception as e:
                    print(f"Error writing {name}: {e}")
                    raise Exception
            else:
                print(f"{dataset_path} not exist in root")
                raise NotImplementedError       

print("All data has been processed, merged, and saved.")