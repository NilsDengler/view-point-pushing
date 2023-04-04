import sys, os
import matplotlib.pyplot as plt
default_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(default_path + 'network/')
sys.path.append(default_path + 'utils/')
sys.path.append(default_path + 'environment/')
sys.path.append(default_path + 'networks/vae/models/')
sys.path.append(default_path)
sys.path.append(os.path.dirname(__file__))
from main_environment import MainEnv
from mapping import ScanningTask
from rl_network_utils import preprocess_images_np, preprocess_images_torch
import torch
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
import gym, shelf_gym
from torch_model_vae import VariationalAutoencoder, CustomDataset
from collections import deque
from dataset_utils import np_to_h5
from tqdm import tqdm
from vpp import VewPointPlanner

class VewPointTask(MainEnv):
    def __init__(self, render=False, shared_memory=False, hz=240, use_egl=False, use_qmap=False):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz, use_egl=use_egl)
        # set global parameters
        self.entropy_change = 0
        self.current_steps = 0
        self.local_steps = 0
        self.kommulative_reward = 0
        self.max_repeat_reached = 0
        self.max_repeat_steps = 10
        self.max_steps = 10
        self.success_thresh = 0.20
        self.global_step = 0
        # set action and observation space
        self._set_action_space()
        self._set_observation_space()
        # load vae and vpp sub-module
        self.vae = self._setup_vae()
        self.vpp = VewPointPlanner(self)
        self.scanning = ScanningTask(self)
        # define workspace bounds
        self.wb = np.array([[-0.34, 0.34], [0.45, 0.7], [self.default_z, self.default_z + 0.2], [-15, 15], [0, 20]]) #workspace bounds
        self.ob = np.array([[-0.34, 0.34], [0.78, 0.95]]) # object placing bounds
        self.object_number = 10
        self.entropy_history = deque(self.max_repeat_steps * [1.], self.max_repeat_steps)
        self.sample_obj_num = True
        self.world_id = self._p.saveState()
        self.is_reset = False


    def _setup_vae(self):
        self.torch_device = torch.device("cuda:0")
        saved_model = os.path.join(os.path.dirname(__file__),
                                   '../networks/vae/saved_models/example_vae')
        vae_prob_map = VariationalAutoencoder(32, 0.5, 300, 400)
        if torch.cuda.is_available():
            vae_prob_map.load_state_dict(torch.load(saved_model))
        else:
            vae_prob_map.load_state_dict(torch.load(saved_model, map_location=torch.device('cpu')))
        vae_prob_map = vae_prob_map.to(self.torch_device)
        vae_prob_map.eval()
        return vae_prob_map


    def _set_action_space(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(5,), dtype=np.float32)# x, y, z, angle


    def _set_observation_space(self):
        self.observation_space = gym.spaces.Box(-np.ones(32 + 11), np.ones(32 + 11), dtype=np.float32)


    def reset(self, evaluation=False, saved_obj_config=None):
        self.is_reset = True
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        # reset scanning variables
        self.scanning.reset_scanning()
        # reset target and obstacle to initial position
        if evaluation:
            self.current_obj_ids = self.obj.reset_objects_fix(saved_obj_config)
        else:
            self.current_obj_ids = self.obj.reset_objects_for_rl(self.ob, self.sample_obj_num, self.object_number)
        self.step_simulation(self.per_step_iterations)
        # initialize object infos
        self.initialize_object_info()
        # save initial object position and check if contact or tilted
        obj_tilt_list, _, obj_contact_list = self.check_object_interaction(tilt=True, update=True, contact=True)
        # generate ground truth map
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        return self.vpp._reset(obj_tilt_list.any(), obj_contact_list.any())


    def step(self, action):
        if self.is_reset:
            self.current_steps, self.local_steps, self.kommulative_reward = 0, 0, 0
            self.is_reset = False
        self.local_steps += 1
        self.current_steps += 1
        self.global_step += 1
        # perform one vpp step.
        # Output :
        # observation, reward, prob_map, target_pos, info_gain, motion_cost, is_success, tilted,
        # contact, max_iter, term, entropy_history
        step_data = self.vpp._step(action, self.wb, self.current_steps, self.entropy_history)
        self.kommulative_reward += step_data[1]
        self.prob_map = step_data[2]
        self.entropy_history = step_data[-1]
        self.termination_info = step_data[-2]
        self.set_var_for_callback(step_data)
        info = {'is_success': step_data[6],
                'is_collision': step_data[7] or step_data[8]}
        return step_data[0], step_data[1], self.termination_info, info


################################
    '''miscellaneous'''
################################
    def set_var_for_callback(self, data):
        self.max_step_reached = data[9]
        self.tilt_check = data[7]
        self.contact_check = data[8]
        self.is_success = data[6]


    def initialize_object_info(self):
        self.current_obst_pos = np.zeros((len(self.current_obj_ids), 6))
        self.object_tilted = np.zeros((len(self.current_obj_ids)))
        self.object_contact = np.zeros((len(self.current_obj_ids)+1))

    def update_object_info(self, obj_id, idx):
        pos, orn = self._p.getBasePositionAndOrientation(obj_id)
        self.current_obst_pos[idx][:3] = pos
        self.current_obst_pos[idx][3:] = self._p.getEulerFromQuaternion(orn)
        return self.current_obst_pos

    def check_for_tilted(self, obj_id, idx):
        self.object_tilted[idx] = self.obj.check_object_drop(obj_id, reset=False)
        return self.object_tilted

    def check_for_contact(self, obj_id, idx):
        contact, close = [], []
        for j in self.arm_joint_indices:
            contact, close = self.col.collision_checks(contact, close, self.robot_id, obj_id, link_A=j)
            self.object_contact[idx] = True if contact or close else False
        return self.object_contact

    def check_object_interaction(self, contact=True, tilt=True, update=True):
        is_tilted, is_update, has_contact = [], [], []
        for i, obj_id in enumerate(list(self.current_obj_ids)+[self.shelf_id]):
            if obj_id != self.shelf_id:
                if tilt: is_tilted = self.check_for_tilted(obj_id, i)
                if update: is_update = self.update_object_info(obj_id, i)
            if contact: has_contact = self.check_for_contact(obj_id, i)
        return is_tilted, is_update, has_contact


    def collect_data(self, save_path):
        self.max_steps = 250
        self.max_repeat_steps = 5
        sample_size = self.max_steps * 6
        print("collect sampled data for ", sample_size, " steps.")
        batches = 22
        pbar = tqdm(total=batches*sample_size)
        for n in range(batches):
            np_samples = np.zeros((sample_size, 300, 400)).astype(np.float16)
            _ = self.reset()
            d = False
            for i in range(sample_size):
                if d: self.reset()
                o, r, d, _ = self.step(self.action_space.sample())
                np_samples[i] = preprocess_images_np(self.scanning.cropped_prob_map)
                pbar.update(1)
            np_to_h5(save_path+str(n)+".h5", np_samples)
        self.close()

################################
    '''main'''
################################


def load_default_task():
    env = make_vec_env('ViewPointNewEnvGUI-v0', n_envs=1)
    return env.envs[0].env.env

def test_env(environment):
    environment.reset()
    while True:
        action = environment.action_space.sample()
        _, r, _, _ = environment.step(action)


if __name__ == '__main__':
    environment = load_default_task()
    environment.collect_data(os.path.join(os.getcwd(), '../networks/vae/train_data/sim_data_300_400_1500'), use_trained=True)
    #collect_data(environment)
    #test_env(environment)

