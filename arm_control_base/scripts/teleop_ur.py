import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time
from pathlib import Path
import cv2
import numpy as np
import json
import logging
from scipy.spatial.transform import Rotation
import multiprocessing
import hydra
from omegaconf import DictConfig
import socket

# third-party / project-local
from io_devices.meta_quest2 import Meta_quest2
from cam_base.camera_redis_interface import CameraRedisSubInterface
from utils import YamlConfig

# Universal robot api
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface
# from ur_rtde import RTDEControlInterface, RTDEReceiveInterface, DashboardClient # ur_rtde

# ------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------
# Function to power on and release the robot via the dashboard server
def dashboard_to_running(ip: str, port: int = 29999):
    """Power-on and brake-release the robot via the dashboard server (port 29999)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as dash:
        dash.settimeout(2.0)
        dash.connect((ip, port))
        dash.sendall(b"power on\n")
        time.sleep(0.5)
        dash.sendall(b"brake release\n")
        time.sleep(0.5)


# ---------------- Robotiq via socket ------------------------------------------------
def rq_connect(ip: str, port: int = 63352):
    """Open persistent TCP socket to Robotiq URCap server."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    s.connect((ip, port))
    # Activate gripper (if not already): ACT=1, GTO=1
    s.sendall(b"SET ACT 1\n")
    s.sendall(b"SET GTO 1\n")
    time.sleep(0.1)
    return s
# Function to set the gripper position via socket
def rq_set_pos(sock: socket.socket, pos: int):
    """Send position (0‑255)."""
    cmd = f"SET POS {int(pos)}\n".encode()
    sock.sendall(cmd)
# Convenience wrappers
rq_open  = lambda sock: rq_set_pos(sock, 0)
rq_close = lambda sock: rq_set_pos(sock, 255)


# Function to convert input from the device to an action for the robot arm
def input2action(device, controller_type="cartesian_servo_position", gripper_dof=1):
    """Convert Meta-Quest pose → [x y z rx ry rz grasp]. """
    state = device.get_controller_state()
    assert state, "please check your headset if works on debug mode correctly"
    
    # Get the target pose matrix, grasp state, reset state, action hot vector, and stop record flag
    target_pose_mat, grasp, stop, action_hot, over = state["target_pose"], state["grasp"], state["stop"], state["action_hot"], state["over"]

    # Get target position
    target_pos = target_pose_mat[:3, 3:]
    action_pos = target_pos.flatten() * 1

    # Get target rotation   
    target_rot = target_pose_mat[:3, :3]   # @ reset_coord_mat[:3,:3] # Reset coordiate
    target_rot_mat = Rotation.from_matrix(target_rot)

    # Convert rotation to Euler angles for the action
    axis_angle = target_rot_mat.as_rotvec(degrees=False) # axis angles in radian
    action_axis_angle = axis_angle.flatten() * 1

    if controller_type == "cartesian_servo_position":
        grasp_val = 1 if grasp else -1
        action = action_pos.tolist() + action_axis_angle.tolist() + [grasp_val] 
    else:
        raise NotImplementedError(f"Controller type {controller_type} is not implemented")

    return action, grasp_val, action_hot, stop, over

# # Safety control function to ensure the target action is within the robot's operational limits
# def safety_control(target_action):
#     if not 0.28 <= target_action[0] <= 0.79:
#         return False
#     if not -0.3 <= target_action[1] <= 0.3:
#         return False
#     if not 0.0 <= target_action[2]:
#         return False
#     return True


# ------------------------------------------------------------------
# Teleoperation + data collection
# ------------------------------------------------------------------
class RtdeDataCollection():
    def __init__(self,
                 observation_cfg,
                 robot_type="ur30",
                 controller_type="cartesian_servo",
                 folder="data_ur30",
                 max_steps=10000,
                 save2memory_first=False,
                 control_frequency=50,
                 FT_option=True,
                 cam_necessary=True,
                 demo_save_frequency=10,
                 ):
                self.robot_type = robot_type
                self.folder = Path(folder)
                self.controller_type = controller_type
                self.observation_cfg = observation_cfg
                self.camera_ids = observation_cfg["camera_ids"]
                self.camera_names = observation_cfg["camera_names"]
                self.save2memory_first = save2memory_first
                self.control_frequency = control_frequency
                self.FT_option = FT_option
                self.cam_necessary = cam_necessary
                self.demo_save_frequency = demo_save_frequency

                self.folder.mkdir(parents=True, exist_ok=True)
                logging.basicConfig(level=logging.INFO)
                self.logger = logging.getLogger("URDataCollection")
                self.logger.info(f"Will save to {self.folder}")
                self.max_steps = max_steps
                self.tmp_folder = None 

                self.obs_action_data = {
                    "action": [],
                    "ee_states": [],
                    "joint_states": [],
                    "gripper_states": [],
                    "action_hot": [],
                    "action_grasp": [],
                    "FT_raw": [],
                    "FT_processed": [],
                }
    
    # ------------------------------------------------------------------
    # Main routine
    # ------------------------------------------------------------------
    def collect_data(self, ip: str):
        # Initialize camera subscribers (optional) 
        if self.cam_necessary:
            cam_interfaces = {}
            for idx, camera_id in enumerate(self.camera_ids):
                camera_info = {"camera_id": camera_id, "camera_name": self.camera_names[idx]}
                cam_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False)
                cam_interface.start()
                cam_interfaces[camera_id] = cam_interface
                self.obs_action_data[f"camera_{camera_id}"] = []
        
        # Initialize the robot and Bring robot to RUNNING (dashboard)
        dashboard_to_running(ip)
        # RTDE control / receive handles
        rtde_c = RTDEControlInterface(ip)
        rtde_r = RTDEReceiveInterface(ip)
        
        # Initialize the FT sensor and Calibrate force sensor (use RTDE to control)
        # inside sensor is always online, no need to enable it
        if self.FT_option:
            rtde_c.zeroFtSensor()
            time.sleep(0.2)

        # gripper socket 
        rq_sock = rq_connect(ip)

        # Initialize the metaquest2
        # seed Meta Quest with current pose (axis‑angle)
        device = Meta_quest2()
        arm_init_pose = np.asarray(rtde_r.getActualTCPPose())  # [x, y, z, rx, ry, rz]
        arm_init_pose_ = Rotation.from_rotvec(arm_init_pose[3:]).as_matrix() # 3x3 rotation matrix
        arm_init_pose_mat = np.eye(4)
        arm_init_pose_mat[:3, :3] = arm_init_pose_
        arm_init_pose_mat[:3, 3] = arm_init_pose[:3]
        device.start_control(arm_init_pose_mat)
        time.sleep(0.5)

        # runtime vars
        i = 0
        gripper_state = 1 # -1 for open, 1 for close #TODO: Tao: HARDCORD here
        # start = False
        # main control‑and‑log loop
        while i < self.max_steps:
            i += 1
            start_time = time.time_ns()
            
            # if action is None and start:
            #     device.stop_control()
            #     break
            # start = True

            # log at reduced rate
            if i % (self.control_frequency / self.demo_save_frequency) == 0:
                # images: Load the observation data into CPU memory
                for camera_id in self.camera_ids:
                    img_info = cam_interfaces[camera_id].get_img_info()
                    imgs_array = cam_interfaces[camera_id].get_img()
                    if cam_interfaces[camera_id].use_color:
                        img_bgr = cv2.cvtColor(imgs_array["color"], cv2.COLOR_RGB2BGR)
                        if self.save2memory_first:
                            # saving images into memory first
                            img_info["color_image_data"] = img_bgr
                        else:
                            # saving images into disk
                            success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_bgr)
                            if not success:
                                print(f"Failed to save image for camera {camera_id}")
                        self.obs_action_data[f"camera_{camera_id}"].append(img_info)
                    
                # FT sensor data
                if self.FT_option:
                    raw_wrench = rtde_r.getFtRawWrench()         # raw wrench data (force/torque)
                    ext_wrench = rtde_r.getActualTCPForce()      # TCP external force/torque
                    self.obs_action_data["FT_raw"].append(raw_wrench)
                    self.obs_action_data["FT_processed"].append(ext_wrench)
                    print("Force/Torque values:", ext_wrench)
                    print("Raw Force/Torque values:", raw_wrench) # 6D: [Fx, Fy, Fz, Tx, Ty, Tz]
                else:
                    raise Exception("Failed to get FT sensor data")

                # save states
                ee_state    = rtde_r.getActualTCPPose()   # 6-D TCP pose
                joint_state = rtde_r.getActualQ()         # 6 joint angles

                self.obs_action_data["ee_states"].append(ee_state)
                self.obs_action_data["joint_states"].append(joint_state)
                self.obs_action_data["gripper_states"].append(gripper_state)

            # read action from the device
            action, action_grasp, action_hot, stop_collection, over = input2action(device=device)
            gripper_state = action_grasp # 1 for grasp, 0 for release
            pose = np.asarray(action[:6])  # 6D m+rad
            # low frequency data collection
            if i % (self.control_frequency / self.demo_save_frequency) == 0:
                self.obs_action_data["action_grasp"].append(action_grasp)
                self.obs_action_data["action"].append(action)
                self.obs_action_data["action_hot"].append(action_hot)
                
                # import pdb;pdb.set_trace()
            # gripper control
                if action_grasp == 1:
                    rq_close(rq_sock)
                elif action_grasp == -1:
                    rq_open(rq_sock)
                else:
                    raise ValueError(f"Invalid action_grasp {action_grasp}")


            # TODO: Tao: if we want to collect multiple demos, we could continue implementing the following code
            # self.stop_collection = stop_collection
            # if stop_collection:
            #     print("one demo over, saving data...")
            #     self.save(keep=True, keyboard_ask=True)
            #     break

            if over:
                device.stop_control()
                del device
                break

            # Safety control 
            # if not safety_control(action):
            #     device.stop_control()
            #     print("The ee location is beyond the predefined safety range")
            #     break

            # perform the action
            action = action[:6]  # only take the first 6 elements for xarm
            print(f"Action: {action}, Grasp: {action_grasp}")
            import pdb; pdb.set_trace()
            rtde_c.servoL(
                pose,                           # target pose (x, y, z, rx, ry, rz)
                0.30,                           # speed  (m/s or rad/s)
                0.30,                           # accel  (m/s² or rad/s²)
                1.0 / self.control_frequency,   # time   = 1 / control frequency
                lookahead_time=0.1,
                gain=300
            )
            print(f"[DEBUG] Would send action: {pose}")

            # control frequency control
            elapsed_time = (time.time_ns() - start_time) / 1e9
            ctrl_setp = 1 / self.control_frequency
            if elapsed_time < ctrl_setp:
                time.sleep(ctrl_setp - elapsed_time)

        # ── end of data collection ────────────────────────────────
        # rtde_c.forceModeStop()           #stop force mode 
        # stop the robot control
        print("Stopping the robot control...")
        rtde_c.stopScript()                # stop the script 
        # stop utde receive
        rtde_c.disconnect()
        rtde_r.disconnect()                
        # stop Robotiq gripper socket
        if rq_sock:                        # rq_sock = socket.socket(..., 63352)
            rq_sock.close()
            rtde_c.disconnect()
            rtde_r.disconnect()
        # Robot comes to Idle
        # dashboard = DashboardClient(ip); dashboard.connect(); dashboard.brakeEngage(); dashboard.disconnect()


    # ---------------------Save the collected data---------------------
    def save(self, keep=True, keyboard_ask=False):
        # get the experiment id
        experiment_id = 0   
        for path in self.folder.glob("run*"):
            if not path.is_dir():
                continue
            try:
                folder_id = int(str(path).split("run")[-1])
                if folder_id > experiment_id:
                    experiment_id = folder_id
            except BaseException:
                pass
        experiment_id += 1
        run_folder = self.folder / f"run{experiment_id:03d}"
        run_folder.mkdir(parents=True, exist_ok=True)
        self.tmp_folder = str(run_folder)
    
        os.makedirs(self.tmp_folder, exist_ok=True)
        with open(f"{self.tmp_folder}/config.json", "w") as f:
            config_dict = {
                "controller_type": self.controller_type,
                "observation_cfg": self.observation_cfg,
            }
            json.dump(config_dict, f)
        
        if keyboard_ask:
            valid_input = False
            while not valid_input:
                try:
                    keep = input(f"Save to {self.tmp_folder} or not? (enter 0 or 1): ")
                    keep = bool(int(keep))
                    valid_input = True
                except Exception as e:
                    print("Invalid input, please enter 0 or 1 and press Enter. Error message：", e)

        if not keep:
            import shutil
            shutil.rmtree(f"{self.tmp_folder}")

        else:
            data = self.obs_action_data
            print("Demo downsmapling frequencey: ", self.demo_save_frequency)
            print("Total length of the trajectory: ", len(data["action"]))

            np.savez(f"{self.tmp_folder}/demo_ee_states", data=np.array(data["ee_states"])) 
            np.savez(f"{self.tmp_folder}/demo_joint_states", data=np.array(data["joint_states"]))
            np.savez(f"{self.tmp_folder}/demo_gripper_states", data=np.array(data["gripper_states"]))
            np.savez(f"{self.tmp_folder}/demo_action", data=np.array(data["action"]))
            np.savez(f"{self.tmp_folder}/demo_action_grasp", data=np.array(data["action_grasp"]))
            np.savez(f"{self.tmp_folder}/demo_action_hot", data=np.array(data["action_hot"]))

            if self.FT_option:
                np.savez(f"{self.tmp_folder}/demo_FT_raw", data=np.array(data["FT_raw"]))
                np.savez(f"{self.tmp_folder}/demo_FT_processed", data=np.array(data["FT_processed"]))

            if self.save2memory_first:
                print("------------- Saving images... -------------")
                for camera_id in self.camera_ids:
                    for img_info in data[f"camera_{camera_id}"]:
                        if "color_image_data" in img_info:
                            success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_info["color_image_data"])
                            if not success:
                                print("failed saving imgs")
                            del img_info["color_image_data"]

            for camera_id in self.camera_ids:
                np.savez(f"{self.tmp_folder}/demo_camera_{camera_id}", data=np.array(data[f"camera_{camera_id}"]))

# main function to run the data collection
@hydra.main(version_base=None, config_path="../configs", config_name="data_collection_ur")
def main(cfg: DictConfig):
    observation_cfg = YamlConfig(cfg.obs_cfg).config  
    data_collection = RtdeDataCollection(
        observation_cfg=observation_cfg,
        robot_type=cfg.robot,
        controller_type=cfg.controller_type,
        folder=cfg.dataset_name,
        max_steps=cfg.max_steps,
        save2memory_first=cfg.save2memory_first,
        control_frequency=cfg.control_frequency,
        FT_option=cfg.FT_option,
        cam_necessary=cfg.cam_necessary,
        demo_save_frequency = cfg.demo_save_frequency,
    )
    data_collection.collect_data(ip=cfg.ip)
    data_collection.save(keep=True, keyboard_ask=True)


if __name__ == "__main__":

    main()

    """
    data collection guidance:
    - run the script
    - press A to end teleoperation then start saving data

    """