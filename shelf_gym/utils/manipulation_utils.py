import numpy as np
import math
from math_utils import  rotate
from collections import deque
#import pb_ompl


class Manipulation():
    def __init__(self, sim, p_id):
        self.sim = sim
        self._p = p_id
        self.tcp_id = self.sim.tool_tip_id

    def move_gripper(self, gripper_opening_length):
        gripper_opening_angle = 0.715 - math.asin((gripper_opening_length - 0.010) / 0.1143)  # angle calculation
        self.sim.controlGripper(controlMode=self._p.POSITION_CONTROL,targetPosition=gripper_opening_angle)
        return


    def reset_robot(self, parameters):
        for i, name in enumerate(self.sim.controlJoints):
            joint = self.sim.joints[name]
            if i == 6:
                targetPosition = 0.715 - math.asin((parameters[i] - 0.010) / 0.1143)
                self.sim.controlGripper(controlMode=self._p.POSITION_CONTROL, targetPosition=targetPosition)
                continue
            self._p.resetJointState(self.sim.robot_id, joint.id, parameters[i])
            self._p.setJointMotorControl2(self.sim.robot_id, joint.id, self._p.POSITION_CONTROL,
                                          targetPosition=parameters[i], force=joint.maxForce,
                                          maxVelocity=joint.maxVelocity)
        self.sim.step_simulation(self.sim.per_step_iterations)


    def get_increment_action(self, motion):
        state = self._p.getLinkState(self.sim.robot_id, self.sim.eef_id, computeForwardKinematics=True)
        return np.array(state[0]) + np.array(motion[:3]), \
               np.array(self._p.getEulerFromQuaternion(state[1])) + np.array(motion[3:6])


    def get_ik_joints(self, position, orientation, link):
        joints = self._p.calculateInverseKinematics(self.sim.robot_id, link,
                                                    position, self._p.getQuaternionFromEuler(orientation),
                                                    solver=self._p.IK_DLS, maxNumIterations=1000, residualThreshold=1e-5)
        return list(joints)[:6]


    def send_position_command(self, target_joint_states):
        num_motors = 6
        position_gain = 0.4
        #for i in [1, 2, 3, 4, 5, 6]:
        #    self._p.setJointMotorControl2(self.robot.id, i, self._p.POSITION_CONTROL, i, force=5 * 240.)
        self._p.setJointMotorControlArray(self.sim.robot_id, [1, 2, 3, 4, 5, 6], self._p.POSITION_CONTROL,
                                          target_joint_states, [0.] * num_motors, [150, 150, 150, 28, 28, 28],
                                          [position_gain] * num_motors, [1.] * num_motors)


    def joint_loop(self, pos, orn, perc=0.1, absolute=False, link_id=-1):
        if link_id == -1:
            link_id = self.tcp_id
        state = self._p.getLinkState(self.sim.robot_id, link_id, physicsClientId=self.sim.client_id)
        pos, orn = self.interpolate_position([pos, orn], state, perc)
        #target_joint_states = self.get_ik_fast(np.array([pos, orn]))
        #if target_joint_states is None:
        target_joint_states = np.array(self.get_ik_joints(pos, orn, self.sim.tool_tip_id)) # PB ik solver
        self.send_position_command(target_joint_states)
        if absolute:
            self.carry_out_motion(target_joint_states)
        self.sim.step_simulation(self.sim.per_step_iterations)

    def interpolate_position(self, target_state, orig_state, pos_perc=0.5):
        orn_perc = 1
        interpol_pos = [(1 - pos_perc) * x1 + pos_perc * x2 for x1, x2 in zip(orig_state[0], target_state[0])]
        orig_ori = orig_state[1]
        if len(orig_state[1]) > 3:
            orig_ori = self._p.getEulerFromQuaternion(orig_state[1])
        interpol_ori = [(1 - orn_perc) * x1 + orn_perc * x2 for x1, x2 in zip(orig_ori, target_state[1])]
        return np.array(interpol_pos), np.array(interpol_ori)


    def carry_out_motion(self, target_joint_states, max_it=1000):
        past_joint_pos = deque(maxlen=5)
        joint_state = self._p.getJointStates(self.sim.robot_id, [i for i in range(1,7)])
        joint_pos = list(zip(*joint_state))[0]
        n_it = 0
        while not np.allclose(joint_pos, target_joint_states, atol=1e-3) and n_it < max_it:
            self._p.stepSimulation()
            n_it += 1
            # Check to see if the arm can't move any close to the desired joint position
            if len(past_joint_pos) == 5 and np.allclose(past_joint_pos[-1], past_joint_pos, atol=1e-3):
                break
            past_joint_pos.append(joint_pos)
            joint_state = self._p.getJointStates(self.sim.robot_id, [i for i in range(1,7)])
            joint_pos = list(zip(*joint_state))[0]


    def push_in_steps(self, target_pos, ori_off=np.array([0, 0, 0])):
        state = self._p.getLinkState(self.sim.robot_id, self.sim.tool_tip_id, physicsClientId=self.sim.client_id)
        contact_array = np.zeros_like(self.sim.object_contact)
        for i in range(1, 21):
            pos, orn = self.interpolate_position([target_pos, self.sim.init_ori+ori_off], state, 0.05 * i)
            self.joint_loop(pos, orn, perc=1., absolute=True)
            #check for contact
            for i, obj_id in enumerate(self.sim.current_obj_ids):
                self.sim.check_for_contact(obj_id, i)
            contact_array = contact_array + self.sim.object_contact
        self.sim.object_contact = contact_array


    def perform_push_primitive(self, target_pos):
        ori_off = np.array([0, 0, 0])
        self.push_in_steps(target_pos, ori_off)
        contact_1 = self.sim.object_contact.copy()

        new_pose = np.array(self._p.getLinkState(self.sim.robot_id, self.sim.tool_tip_id)[0])
        new_pose[:2] = rotate(new_pose[:2] + np.array([0, target_pos[4]]), target_pos[3], origin=new_pose[:2])
        self.push_in_steps(new_pose, ori_off)
        contact_2 = self.sim.object_contact.copy()

        self.push_in_steps(self.sim.init_pos, ori_off)
        self.push_in_steps(self.sim.init_pos)

        self.sim.object_contact = contact_1 + contact_2
        return new_pose


    def perform_ompl(self, position, orientation):
        #set start state
        start_joint_states = self.sim.get_current_joint_config()
        self.ompl_robot.set_state(start_joint_states)
        #calculate target position
        pos = position
        orn = self.sim.init_ori+orientation
        #perform ik for goal state
        target_joint_states = list(self.get_ik_joints(pos, orn, self.sim.tool_tip_id))
        if not target_joint_states: return False
        #plan path from start to goal with ompl
        res, path = self.pb_ompl_interface.plan(target_joint_states, allowed_time=0.5)
        if not res:
            return False
        #if there is a path, execute it
        self.pb_ompl_interface.execute(path)
        return True


    def create_ompl(self):
        # setup pb_ompl
        self.ompl_robot = pb_ompl.PbOMPLRobot(self.sim.robot_id)
        self.pb_ompl_interface = pb_ompl.PbOMPL(self.ompl_robot)
        self.pb_ompl_interface.set_planner("BITstar")
        #self.ompl_robot.set_state(self.sim.initial_parameters[:-1])
        self.pb_ompl_interface.set_obstacles([self.sim.UR5Stand_id, self.sim.shelf_id, self.sim.wall_id])


    def move_to_viewpoint(self, target_pos):
        ori_off = np.array([(target_pos[4]*np.pi)/180, 0,  (target_pos[3] * np.pi)/180])
        # using internal pybullet function
        #debug_point = SphereMarker(target_pos[:3], p_id=self.sim._p)
        #print("wanted pos:", target_pos[:3])
        #self.push_in_steps(target_pos[:3], ori_off)
        #print("tcp: ", self.sim._p.getLinkState(self.sim.robot_id, self.sim.tool_tip_id, physicsClientId=self.sim.client_id)[0])
        # using ompl
        #self.create_ompl()
        #done = self.perform_ompl(target_pos[:3], ori_off)
        #if not done:
        #    self.push_in_steps(target_pos[:3], ori_off)
        #using only entpoints and resetting ee
        self.fast_vp(target_pos)


    def fast_vp(self, tp):
        orn = self.sim.init_ori + np.array([(tp[4] * np.pi) / 180, 0, (tp[3] * np.pi) / 180])
        #debug_point = SphereMarker(tp[:3], p_id=self.sim._p)
        target_joint_states = self.get_ik_joints(tp[:3], orn, self.sim.tool_tip_id)
        target_joint_states.append(0)
        self.reset_robot(target_joint_states)
