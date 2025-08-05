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
import hydra
from omegaconf import DictConfig

# third-party / project-local
from io_devices.meta_quest2 import Meta_quest2
from cam_base.camera_redis_interface import CameraRedisSubInterface
from utils import YamlConfig

# ---------------- MOCK CLASSES ----------------
class MockRTDEControlInterface:
    def zeroFtSensor(self): print("[MOCK] zeroFtSensor called")
    def stopScript(self): print("[MOCK] stopScript called")
    def disconnect(self): print("[MOCK] disconnect called")
    def servoL(self, *args, **kwargs): print(f"[MOCK] servoL called with args: {args}, kwargs: {kwargs}")

class MockRTDEReceiveInterface:
    def getActualTCPPose(self): return [0.5, 0, 0.3, 0, 0, 0]
    def getActualQ(self): return [0, 0, 0, 0, 0, 0]
    def getFtRawWrench(self): return [0, 0, 0, 0, 0, 0]
    def getActualTCPForce(self): return [0, 0, 0, 0, 0, 0]
    def disconnect(self): print("[MOCK] disconnect called")

def dashboard_to_running(ip: str, port: int = 29999):
    print(f"[MOCK] Would connect to dashboard at {ip}:{port} and power on + brake release.")

def rq_connect(ip: str, port: int = 63352):
    print(f"[MOCK] Would connect to Robotiq gripper at {ip}:{port}")
    return "mock_socket"
def rq_set_pos(sock, pos: int):
    print(f"[MOCK] Would set gripper pos to {pos}")
rq_open  = lambda sock: rq_set_pos(sock, 0)
rq_close = lambda sock: rq_set_pos(sock, 255)

def input2action(device, controller_type="cartesian_servo_position", gripper_dof=1):
    # 这里直接调用真实的 Meta_quest2 逻辑
    state = device.get_controller_state()
    assert state, "please check your headset if works on debug mode correctly"
    target_pose_mat, grasp, stop, action_hot, over = state["target_pose"], state["grasp"], state["stop"], state["action_hot"], state["over"]
    target_pos = target_pose_mat[:3, 3:]
    action_pos = target_pos.flatten() * 1
    target_rot = target_pose_mat[:3, :3]
    target_rot_mat = Rotation.from_matrix(target_rot)
    axis_angle = target_rot_mat.as_rotvec(degrees=False)
    action_axis_angle = axis_angle.flatten() * 1
    if controller_type == "cartesian_servo_position":
        grasp_val = 1 if grasp else -1
        action = action_pos.tolist() + action_axis_angle.tolist() + [grasp_val] 
    else:
        raise NotImplementedError(f"Controller type {controller_type} is not implemented")
    return action, grasp_val, action_hot, stop, over

class RtdeDataCollection():
    def __init__(self,
                 observation_cfg,
                 robot_type="ur30",
                 controller_type="cartesian_servo",
                 folder="data",
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
    
    def collect_data(self, ip: str):
        # 相机采集可用时保留，否则可注释
        if self.cam_necessary:
            cam_interfaces = {}
            for idx, camera_id in enumerate(self.camera_ids):
                camera_info = {"camera_id": camera_id, "camera_name": self.camera_names[idx]}
                cam_interface = CameraRedisSubInterface(camera_info=camera_info, use_depth=False)
                cam_interface.start()
                cam_interfaces[camera_id] = cam_interface
                self.obs_action_data[f"camera_{camera_id}"] = []
        else:
            cam_interfaces = {}

        # MOCK机械臂和夹爪
        dashboard_to_running(ip)
        rtde_c = MockRTDEControlInterface()
        rtde_r = MockRTDEReceiveInterface()
        rq_sock = rq_connect(ip)

        # Meta Quest 2
        device = Meta_quest2()
        tcp_init_pose = np.asarray(rtde_r.getActualTCPPose())
        rot_init_mat = Rotation.from_rotvec(tcp_init_pose[3:]).as_matrix()
        tcp_init_mat = np.eye(4)
        tcp_init_mat[:3, :3] = rot_init_mat
        tcp_init_mat[:3, 3] = tcp_init_pose[:3]
        device.start_control(tcp_init_mat)
        time.sleep(1.0)

        i = 0
        gripper_state = 1
        while i < self.max_steps:
            i += 1
            start_time = time.time_ns()
            if self.cam_necessary and i % (self.control_frequency / self.demo_save_frequency) == 0:
                for camera_id in self.camera_ids:
                    img_info = cam_interfaces[camera_id].get_img_info()
                    imgs_array = cam_interfaces[camera_id].get_img()
                    if cam_interfaces[camera_id].use_color:
                        img_bgr = cv2.cvtColor(imgs_array["color"], cv2.COLOR_RGB2BGR)
                        if self.save2memory_first:
                            img_info["color_image_data"] = img_bgr
                        else:
                            success = cv2.imwrite(img_info["color_img_name"] + ".jpg", img_bgr)
                            if not success:
                                print(f"Failed to save image for camera {camera_id}")
                        self.obs_action_data[f"camera_{camera_id}"].append(img_info)
                    if self.FT_option:
                        raw_wrench = rtde_r.getFtRawWrench()
                        ext_wrench = rtde_r.getActualTCPForce()
                        self.obs_action_data["FT_raw"].append(raw_wrench)
                        self.obs_action_data["FT_processed"].append(ext_wrench)
                        print("Force/Torque values:", ext_wrench)
                        print("Raw Force/Torque values:", raw_wrench)
            if self.cam_necessary and i % (self.control_frequency / self.demo_save_frequency) == 0:
                ee_state = rtde_r.getActualTCPPose()
                joint_state = rtde_r.getActualQ()
                self.obs_action_data["ee_states"].append(ee_state)
                self.obs_action_data["joint_states"].append(joint_state)
                self.obs_action_data["gripper_states"].append(gripper_state)

            action, action_grasp, action_hot, stop_collection, over = input2action(device=device)
            gripper_state = action_grasp
            pose = np.asarray(action[:6])
            if i % (self.control_frequency / self.demo_save_frequency) == 0:
                self.obs_action_data["action_grasp"].append(action_grasp)
                self.obs_action_data["action"].append(action)
                self.obs_action_data["action_hot"].append(action_hot)
                # MOCK gripper
                if action_grasp == 1:
                    rq_close(rq_sock)
                elif action_grasp == -1:
                    rq_open(rq_sock)
                else:
                    raise ValueError(f"Invalid action_grasp {action_grasp}")

            if over:
                device.stop_control()
                del device
                break

            # MOCK机械臂动作
            print(f"[MOCK] Would send action: {pose}")

            elapsed_time = (time.time_ns() - start_time) / 1e9
            ctrl_setp = 1 / self.control_frequency
            if elapsed_time < ctrl_setp:
                time.sleep(ctrl_setp - elapsed_time)

        print("Stopping the robot control...")
        rtde_c.stopScript()
        rtde_c.disconnect()
        rtde_r.disconnect()
        if rq_sock:
            print("[MOCK] Would close gripper socket")

    def save(self, keep=True, keyboard_ask=False):
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