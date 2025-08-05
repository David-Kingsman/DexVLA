from abc import abstractmethod
from typing import Dict, Protocol
import numpy as np

class Robot(Protocol):
    """Robot protocol.
    A protocol for a robot that can be controlled.
    """
    @abstractmethod
    def num_dofs(self) -> int:
        raise NotImplementedError
    @abstractmethod
    def get_joint_state(self) -> np.ndarray:
        raise NotImplementedError
    @abstractmethod
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        raise NotImplementedError
    @abstractmethod
    def command_eef_pose(self, eef_pose: np.ndarray) -> None:
        raise NotImplementedError
    @abstractmethod
    def get_observations(self) -> Dict[str, np.ndarray]:
        raise NotImplementedError

class PrintRobot(Robot):
    def __init__(self, num_dofs: int, dont_print: bool = False):
        self._num_dofs = num_dofs
        self._joint_state = np.zeros(num_dofs)
        self._dont_print = dont_print
    def num_dofs(self) -> int:
        return self._num_dofs
    def get_joint_state(self) -> np.ndarray:
        return self._joint_state
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        assert len(joint_state) == (self._num_dofs), (
            f"Expected joint state of length {self._num_dofs}, "
            f"got {len(joint_state)}."
        )
        self._joint_state = joint_state
        if not self._dont_print:
            print(self._joint_state)
    def command_eef_pose(self, eef_pose: np.ndarray) -> None:
        if not self._dont_print:
            print(f"EEF pose command: {eef_pose}")
    def get_observations(self) -> Dict[str, np.ndarray]:
        joint_state = self.get_joint_state()
        pos_quat = np.zeros(7)
        return {
            "joint_positions": joint_state,
            "joint_velocities": joint_state,
            "ee_pos_quat": pos_quat,
            "gripper_position": np.array(0),
        }

class BimanualRobot(Robot):
    def __init__(self, robot_l: Robot, robot_r: Robot):
        self._robot_l = robot_l
        self._robot_r = robot_r
    def num_dofs(self) -> int:
        return self._robot_l.num_dofs() + self._robot_r.num_dofs()
    def get_joint_state(self) -> np.ndarray:
        return np.concatenate(
            (self._robot_l.get_joint_state(), self._robot_r.get_joint_state())
        )
    def command_joint_state(self, joint_state: np.ndarray) -> None:
        self._robot_l.command_joint_state(joint_state[: self._robot_l.num_dofs()])
        self._robot_r.command_joint_state(joint_state[self._robot_l.num_dofs() :])
    def command_eef_pose(self, eef_pose: np.ndarray) -> None:
        self._robot_l.command_eef_pose(eef_pose[: self._robot_l.num_dofs()])
        self._robot_r.command_eef_pose(eef_pose[self._robot_l.num_dofs() :])
    def get_observations(self) -> Dict[str, np.ndarray]:
        l_obs = self._robot_l.get_observations()
        r_obs = self._robot_r.get_observations()
        assert l_obs.keys() == r_obs.keys()
        return_obs = {}
        for k in l_obs.keys():
            try:
                return_obs[k] = np.concatenate((l_obs[k], r_obs[k]))
            except Exception as e:
                print(e)
                print(k)
                print(l_obs[k])
                print(r_obs[k])
                raise RuntimeError()
        return return_obs

def main():
    pass

if __name__ == "__main__":
    main()