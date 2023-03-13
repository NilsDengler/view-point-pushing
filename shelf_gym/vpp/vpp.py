import numpy as np
from math_utils import translate_to_other_boundries
from collections import deque

class VewPointPlanner():
    def __init__(self, sim):
        self.sim = sim
        self._p = sim._p
        self.vae = sim.vae
        self.entropy_change_history = deque(3 * [1.], 3)

    def _reset(self, is_tilted, has_contact):
        #get initial map from fixed points left, right, center
        gt_target_point = [[-0.25, 0.45, 0.97 + 0.25, -15, 20], [0.25, 0.45, 0.97 + 0.25, 15, 20]]
        self.gt_map = self.get_intial_map(gt_target_point)
        _, self.gt_entropy, _ = self.sim.scanning.compare_to_gt(self.gt_map, self.sim.scanning.initial_prob_map)
        _, _, _, biggest_cp_u = self.sim.scanning.find_center_point(self.gt_map, False)
        return self._get_observation(self.gt_map, last_target_pos=[0.00, 0.49, 0.97, 0, 0], colission_check=is_tilted,
                                     contact_check=has_contact, info_gain=0, motion_cost=0,
                                     unknown_area_center=biggest_cp_u)[0]

    def _get_observation(self, prob_map, last_target_pos, colission_check, contact_check,
                         info_gain, motion_cost, unknown_area_center):
        observation = []
        #get latent space
        latent = self.sim.scanning.get_latent_space(prob_map, self.vae)
        observation.extend(latent)
        #get last action [x,y,z, yaw]
        observation.extend(last_target_pos)
        #get current entropy
        observation.extend([info_gain])
        #get motion cost
        observation.extend([motion_cost])
        #check if collssion or dropp
        observation.extend([int(colission_check or contact_check)])
        #get enter of biggest unknown area
        observation.extend(unknown_area_center)
        return observation, latent


    def _get_reward(self, is_tilt, has_contact, information_gain, motion_cost, alpha=10, beta=0.):
        if is_tilt or has_contact:
            return -25
        return (beta * - motion_cost) + (alpha*information_gain)


    def _get_termination_criteria(self, current_iter, obj_tilt_list, obj_contact_list):
        # check if max steps reached
        max_iter_reached = current_iter >= self.sim.max_steps
        # colission check
        is_tilted = obj_tilt_list.any()
        has_contact = obj_contact_list.any()
        # check for termination criteria success
        if current_iter < 4:
            is_termination = is_tilted or has_contact or max_iter_reached
        else:
            is_termination = is_tilted or has_contact or max_iter_reached or \
                             np.sum(self.entropy_change_history) <= 0.05 or\
                             (np.array(self.entropy_change_history) <= 0.01).all()#or info_gain < 0.005
        #print(info_gain, entropy_change, entropy_change < 0.01, is_tilted, has_contact, max_iter_reached)
        is_success = True
        return is_success, is_tilted, has_contact, max_iter_reached, is_termination


    def _perform_action(self, action, action_bounds):
        target_pos = []
        # translate action from [-1,1] to real borders
        for idx, b in enumerate(action_bounds):
            target_pos.append(translate_to_other_boundries(-1, 1, b[1], b[0], action[idx]))
        #calculate motion cost between current position and target position
        motion_cost = np.linalg.norm(np.array(target_pos[:3]) - np.array(self.sim.get_current_tcp()[0]))
        # move to the calculatetd view point and perform action
        self.sim.man.move_to_viewpoint(target_pos)  # left = -1 , right = 1
        # save current view point
        current_state = self.sim.get_current_tcp()
        current_tcp = [list(current_state[0]), list(self._p.getEulerFromQuaternion(current_state[1]))]
        return target_pos, motion_cost, current_tcp


    def _step(self, action, action_bounds, current_iter, entropy_history):
        #perform the given action
        target_pos, motion_cost, current_tcp = self._perform_action(action, action_bounds)

        #generate the current probability map
        prob_map = self.sim.scanning.get_prob_map()
        test_img, obj_corners, unkn_area_cp, biggest_cp_u = self.sim.scanning.find_center_point(prob_map, True)
        info_gain, entropy_history, self.entropy_change_history = self.sim.scanning.vp_information_gain_calculation(entropy_history, self.entropy_change_history)

        # save current object position and check if contact or tilt
        obj_tilt_list, _, _ = self.sim.check_object_interaction(tilt=True, update=True)

        #check for termination criteria
        is_success, tilted, contact, max_iter, term = self._get_termination_criteria(current_iter, obj_tilt_list,
                                                                                     self.sim.object_contact)
        #get observation
        observation = self._get_observation(prob_map, target_pos, tilted, contact, info_gain, motion_cost, biggest_cp_u)[0]

        # calculate the reward
        reward = self._get_reward(tilted, contact, info_gain, motion_cost)

        return observation, reward, prob_map, target_pos, info_gain, motion_cost, is_success, tilted, \
               contact, max_iter, term, entropy_history


    def get_intial_map(self, target_point):
        prob_map = self.sim.scanning.get_prob_map()
        for tp in target_point:
            self.sim.man.fast_vp(tp)
            prob_map = self.sim.scanning.get_prob_map()
        self.sim.man.reset_robot(self.sim.initial_parameters)
        return prob_map
