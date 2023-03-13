import numpy as np

class Debug():
    def __init__(self, sim, p_id):
        self.sim = sim
        self._p = p_id
        self.goal_point_id = -1

    def set_debug_parameter(self):
        self.xin = self._p.addUserDebugParameter('x', -0.4, 0.4, 0)
        self.yin = self._p.addUserDebugParameter('y', 0.3, 0.9, 0.49)
        self.zin = self._p.addUserDebugParameter('z', 0.8, 1.3, 0.85)
        self.rollId = self._p.addUserDebugParameter(
            'roll', -3.14, 3.14, np.pi / 2)  # -1.57 yaw
        self.pitchId = self._p.addUserDebugParameter(
            'pitch', -3.14, 3.14, 0.0)
        self.yawId = self._p.addUserDebugParameter(
            'yaw', -np.pi , 2*np.pi , np.pi)  # -3.14 pitch
        self.gripper_opening_length_control = self._p.addUserDebugParameter(
            'gripper_opening_length', 0, 0.1, 0.085)

    def read_debug_parameter(self):
        # read the value of task parameter
        x = self._p.readUserDebugParameter(self.xin)
        y = self._p.readUserDebugParameter(self.yin)
        z = self._p.readUserDebugParameter(self.zin)
        roll = self._p.readUserDebugParameter(self.rollId)
        pitch = self._p.readUserDebugParameter(self.pitchId)
        yaw = self._p.readUserDebugParameter(self.yawId)
        gripper_opening_length = self._p.readUserDebugParameter(self.gripper_opening_length_control)
        ls = self._p.getJointStates(self.sim.robot_id, range(0, 7))
        joints = []
        for i in ls:
            joints.append(i[0])
        #print(joints)
        return x, y, z, roll, pitch, yaw, gripper_opening_length

    def set_debug_parameters_joints(self):
        self.j1 = self._p.addUserDebugParameter('j1', -1.57, 1.57, -1.487)
        self.j2 = self._p.addUserDebugParameter('j2', -1.57, 1.57, -0.909)
        self.j3 = self._p.addUserDebugParameter('j3', -1.57, 1.57, 0.826)
        self.j4 = self._p.addUserDebugParameter('j4', -1.57, 1.57, -0.017)
        self.j5 = self._p.addUserDebugParameter('j5', -1.57, 1.57, 1.504)
        self.j6 = self._p.addUserDebugParameter('j6', -1.57, 1.57, 0.00)
        self.gripper_opening_length_control = self._p.addUserDebugParameter(
            'gripper_opening_length', 0, 0.1, 0.085)

    def read_debug_parameter_joints(self):
        # read the value of task parameter
        j1 = self._p.readUserDebugParameter(self.j1)
        j2 = self._p.readUserDebugParameter(self.j2)
        j3 = self._p.readUserDebugParameter(self.j3)
        j4 = self._p.readUserDebugParameter(self.j4)
        j5 = self._p.readUserDebugParameter(self.j5)
        j6 = self._p.readUserDebugParameter(self.j6)
        gripper_opening_length = self._p.readUserDebugParameter(self.gripper_opening_length_control)
        ls = self._p.getLinkState(self.robot_id, self.eef_id)
        return [j1,j2,j3,j4,j5,j6], gripper_opening_length

    def set_debug_camera(self):
        self._p.resetDebugVisualizerCamera(cameraDistance=1.1, cameraYaw=-25.99, cameraPitch=-42.0,
                                           cameraTargetPosition=(0.13026267290115356, 1.1671963930130005, 0.5323477387428284),)

    def visualize_goal_point(self, position):
        if self.goal_point_id <= 0:
            self.goal_point_id = self._p.addUserDebugPoints([position[0]], [[1,0,0]], 5, physicsClientId=self.sim.client_id)
        else:
            self.goal_point_id = self._p.addUserDebugPoints([position[0]], [[1, 0, 0]], pointSize=10.,
                                                            replaceItemUniqueId=self.goal_point_id,
                                                            physicsClientId=self.sim.client_id)


    def draw_sample_area(self, bounds, z, color=[1, 1, 1]):
        self._p.addUserDebugLine([bounds[0][0], bounds[1][0], z], [bounds[0][0], bounds[1][1], z], lineColorRGB=color)
        self._p.addUserDebugLine([bounds[0][0], bounds[1][1], z], [bounds[0][1], bounds[1][1], z], lineColorRGB=color)
        self._p.addUserDebugLine([bounds[0][1], bounds[1][1], z], [bounds[0][1], bounds[1][0], z], lineColorRGB=color)
        self._p.addUserDebugLine([bounds[0][1], bounds[1][0], z], [bounds[0][0], bounds[1][0], z], lineColorRGB=color)

import pybullet as p
import numpy as np

class SphereMarker:
    def __init__(self, position, radius=0.05, rgba_color=(1, 0, 0, 0.8), text=None, orientation=None, p_id=0):
        self.p_id = p_id
        position = np.array(position)
        vs_id = p_id.createVisualShape(
            self.p_id.GEOM_SPHERE, radius=radius, rgbaColor=rgba_color, physicsClientId=0)

        self.marker_id = self.p_id.createMultiBody(
            baseMass=0,
            baseInertialFramePosition=[0, 0, 0],
            baseCollisionShapeIndex=-1,
            baseVisualShapeIndex=vs_id,
            basePosition=position,
            useMaximalCoordinates=False
        )

        self.debug_item_ids = list()
        if text is not None:
            self.debug_item_ids.append(
                self.p_id.addUserDebugText(text, position + radius)
            )

        if orientation is not None:
            # x axis
            axis_size = 2 * radius
            rotation_mat = np.asarray(p.getMatrixFromQuaternion(orientation)).reshape(3, 3)

            # x axis
            x_end = np.array([[axis_size, 0, 0]]).transpose()
            x_end = np.matmul(rotation_mat, x_end)
            x_end = position + x_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, x_end, lineColorRGB=(1, 0, 0))
            )
            # y axis
            y_end = np.array([[0, axis_size, 0]]).transpose()
            y_end = np.matmul(rotation_mat, y_end)
            y_end = position + y_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, y_end, lineColorRGB=(0, 1, 0))
            )
            # z axis
            z_end = np.array([[0, 0, axis_size]]).transpose()
            z_end = np.matmul(rotation_mat, z_end)
            z_end = position + z_end[:, 0]
            self.debug_item_ids.append(
                self.p_id.addUserDebugLine(position, z_end, lineColorRGB=(0, 0, 1))
            )

    def __del__(self):
        self.p_id.removeBody(self.marker_id, physicsClientId=0)
        for debug_item_id in self.debug_item_ids:
            self.p_id.removeUserDebugItem(debug_item_id)