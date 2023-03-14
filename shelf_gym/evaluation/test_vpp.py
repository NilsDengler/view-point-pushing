import sys, os
default_path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(default_path + 'network/')
sys.path.append(default_path + 'utils/')
sys.path.append(default_path + 'task/')
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/LazyThetaStar/build/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../git/ur_ikfast/'))
sys.path.append(default_path + 'environment/')
sys.path.append(default_path +'/networks/vae/models/')
sys.path.append(default_path)
#import for RL
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
import torch
import numpy as np
import time
import gym, shelf_gym


def run_random_vpp(env, max_steps):
    obs = env.reset()
    for idx in range(max_steps):
            action = env.action_space.sample()
            obs, reward, Done, info = env.step(action)
    return env.entropy_history[0]

def run_vpp(env, model):
    obs = env.reset()
    Done = False
    t0 = time.time()
    while not Done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, Done, info = env.step(action)
    time_per_step = (time.time()-t0)/env.current_steps
    return env.entropy_history[0], env.vpp.gt_entropy, env.current_steps, time_per_step

def sample_object_configs(env, sample_num):
    saved_config = []
    for idx in tqdm(range(sample_num)):
        single_sample_config = []
        object_config = 2
        placed_obj_ids = env.obj.reset_objects_for_rl(env.object_bound_list[object_config], env.sample_obj_num,
                                                      env.object_number[object_config])
        for i, obj_id in enumerate(placed_obj_ids):
            pos, orn = env._p.getBasePositionAndOrientation(obj_id)
            single_sample_config.append([obj_id, pos, orn])
        saved_config.append(single_sample_config)
    return saved_config

def generate_five_vp_baseline(env):
    env.scanning.reset_scanning()
    [[-0.25, 0.45, 0.97 + 0.25, -15, 20], [0.25, 0.45, 0.97 + 0.25, 15, 20]]
    target_pos = [[-0.3, 0.55, 0.97, -15, 0], [-0.25, 0.45, 0.97 + 0.25, -15, 20],
                  [0, 0.55, 0.97 + 0.25, 0, 20],[0.25, 0.55, 0.97 + 0.25, 15, 20], [0.3, 0.55, 0.97, 15, 0]]
    five_vp_map = env.vpp.perform_gt_vpp(target_pos)
    _, five_vp_entropy, _ = env.scanning.compare_to_gt(five_vp_map, env.scanning.initial_prob_map)
    return five_vp_entropy

def evaluate_vpp(env, model, eval_num):
    for idx in tqdm(range(eval_num)):
        # run our vpp method
        vpp_entropy, three_vp_entropy, max_steps, time_per_step_vpp = run_vpp(env, model)

        #run random agent
        rnd_agent_entropy = run_random_vpp(env, max_steps)


###### Environmental Functions ######
def load_object_configs(load_path):
    return np.load(load_path, allow_pickle=True)

def set_up_env(env_name, model_name):
    env = make_vec_env(env_name)
    saved_model_path = os.path.join(os.path.dirname(__file__), '../networks/rl/saved_models/' + model_name + "intermediate_saved_model")
    model = TQC.load(saved_model_path)
    return env.envs[0].env.env, model

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    env, model = set_up_env('ViewPointEnvGUI-v0', "TQC_example/model_1/")
    evaluate_vpp(env, model, 10)
