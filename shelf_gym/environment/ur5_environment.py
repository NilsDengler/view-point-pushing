#!/usr/bin/python3
from base_environment import BasePybulletEnv
import pybullet as p
import numpy as np
import os
from robot_utils import setup_sisbot, Robot
from manipulation_utils import Manipulation
#import pb_ompl

class RobotEnv(BasePybulletEnv):
    DEFAULT_PATH = os.path.join(os.path.dirname(__file__), '../')
    def __init__(self,render=False, shared_memory=False, hz=240, use_egl=False):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz, use_egl=use_egl)
        self.gripper_open_limit = (0.0, 0.085)
        self.ee_position_limit = ((-0.4, 0.4), (0.3, 0.9), (0.8, 1.4))
        self.initial_parameters = (1.1297861623268122, -1.4717860300352437, 2.5400780616664034, -1.068276383927301,
                                   1.1306038004860783, 0.0015930864785381922, 0.)

        # create robot entity in environment
        self.load_robot()
        self.joints, self.controlGripper, self.controlJoints, self.mimicParentName = setup_sisbot(p, self.robot_id)

        self.arm_joint_indices = [1, 2, 3, 4, 5, 6, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21]

        self.initialize_gripper()
        self.eef_id = self.joints['ee_fixed_joint'][0]  # ee_link
        self.tool_tip_id = self.joints['tool0_fixed_joint-tool_tip'][0]  # ee_link
        self.camera_link = self.joints['dummy_camera_joint'][0]

        # initialize manipulation class
        self.man = Manipulation(self, self._p)

        # # setup pb_ompl
        # self.ompl_robot = pb_ompl.PbOMPLRobot(self.robot_id)
        # self.pb_ompl_interface = pb_ompl.PbOMPL(self.ompl_robot)
        # self.pb_ompl_interface.set_planner("RRTConnect")
        # self.ompl_robot.set_state(self.initial_parameters[:-1])

        # set robot to intial position
        self.man.reset_robot(self.initial_parameters)
        self.init_pos = list(self._p.getLinkState(self.robot_id, self.tool_tip_id)[0])
        self.init_ori = np.array(self._p.getEulerFromQuaternion(self._p.getLinkState(self.robot_id, self.tool_tip_id)[1]))

    def initialize_gripper(self):
        # Add force sensors
        self._p.enableJointForceTorqueSensor(
            self.robot_id, self.joints['left_inner_finger_pad_joint'].id)
        self._p.enableJointForceTorqueSensor(
            self.robot_id, self.joints['right_inner_finger_pad_joint'].id)

        # Change the friction of the gripper
        self._p.changeDynamics(
            self.robot_id, self.joints['left_inner_finger_pad_joint'].id, lateralFriction=1)
        self._p.changeDynamics(
            self.robot_id, self.joints['right_inner_finger_pad_joint'].id, lateralFriction=1)


    def load_robot(self):
        self.robot_id = self._p.loadURDF(self.DEFAULT_PATH+'meshes/urdf/ur5_robotiq_85_new.urdf' ,[0, 0, 0.9], # urdf_change
                                         self._p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)#,
                                         #flags=self._p.URDF_USE_INERTIA_FROM_FILE)

    def get_current_tcp(self):
        return self._p.getLinkState(self.robot_id, self.tool_tip_id, physicsClientId=self.client_id)

    def get_current_joint_config(self):
        return [self._p.getJointState(self.robot_id, i)[0] for i in range(1, 7)]