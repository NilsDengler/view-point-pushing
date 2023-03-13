import sys, os
default_path = os.path.join(os.path.dirname(__file__), '../../')
sys.path.append(default_path + 'network/')
sys.path.append(default_path + 'utils/')
sys.path.append(default_path + 'vpp/')
sys.path.append(default_path + 'environment/')
sys.path.append(default_path +'/networks/vae/models/')
sys.path.append(default_path)
#import for RL
from sb3_contrib import TQC
from stable_baselines3.common.noise import NormalActionNoise
from custom_callbacks import SavingCallback, ProgressBarManager, TensorboardCallback
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
from rl_network_utils import preprocess_images_np, preprocess_images_torch
from dataset_utils import np_to_h5
import torch
import numpy as np
import gym, shelf_gym

def load_default_task():
    env = make_vec_env('PushEnvGUI-v0', n_envs=1)
    return env.envs[0].env.env

def load_model(algo, env):
    action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape[-1]),
                                     sigma=0.4 * np.ones(env.action_space.shape[-1]))
    policy_kwargs = dict(n_critics=2, activation_fn=torch.nn.LeakyReLU, net_arch=[128, 128, 256])
    print("choosing algorithm: ", algo)
    if algo == "TQC":
        policy = "MlpPolicy"
        training_starts_at = 500
        env.envs[0].env.env.training_starts_at = training_starts_at
        model = TQC(policy, env, tensorboard_log="../tensorboard_logs/", batch_size=256, tau=0.001, learning_rate=1e-5,
                   gamma=0.95, buffer_size=int(1e5), train_freq=(10, 'step'), learning_starts=training_starts_at,
                   top_quantiles_to_drop_per_net=2, gradient_steps=-1,  policy_kwargs=policy_kwargs, action_noise=action_noise,
                   verbose=1, device="cuda")
    return model


def load_train(log_dir_name, tensorboard_log_name, env):
    if os.path.exists(log_dir_name + "intermediate_saved_model.zip"):
        print("LOAD saved model")
        model = TQC.load(log_dir_name + "intermediate_saved_model.zip")
        model.set_env(env)
        if os.path.exists(log_dir_name + "replay_buffer.pkl"):
            model.load_replay_buffer(log_dir_name + "replay_buffer")
    else:  model = load_model("TQC", env)
    max_iterations = 300000
    saving_cb = SavingCallback(log_dir_name, save_freq=500)
    tensorboard_cb = TensorboardCallback()

    try:
        with ProgressBarManager(max_iterations) as prog_cb:
            model.learn(total_timesteps=max_iterations, tb_log_name=tensorboard_log_name,
                        callback=[prog_cb, saving_cb, tensorboard_cb], reset_num_timesteps=False)
    except KeyboardInterrupt:
        pass

    model.save(log_dir_name + "final_model")
    print("saved final model")


def load_test(log_dir_name, env):
    #saved_model_path = log_dir_name + "best_model/best_model"
    saved_model_path = log_dir_name + "intermediate_saved_model"
    #saved_model_path = log_dir_name + "final_model"

    model = TQC.load(saved_model_path)
    eval_num = 200
    for i in tqdm(range(eval_num)):
        obs = env.reset()
        Done = False
        while not Done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, Done, info = env.step(action)


def rl_push(env_name):
    log_dir_name = "saved_models/TQC_example/model_1/"
    tensorboard_log_name = "TQC_example_model_1"

    print("cuda is available: ", torch.cuda.is_available)
    torch.device("cuda")
    torch.set_num_threads(6)

    os.makedirs(log_dir_name, exist_ok=True)

    # Initialize environments
    env = make_vec_env(env_name)

    eval_env = make_vec_env(env_name.replace("GUI", ""))
    #load_train(log_dir_name, tensorboard_log_name, env, eval_env)
    load_test(log_dir_name, env)
    env.envs[0].env.env.close()


def collect_data(env_name):
    h5_save_path = os.path.join(os.path.dirname(__file__), '../../vae/train_data/heightmap_data.h5')
    env = make_vec_env(env_name).envs[0].env.env
    env.sample_data = True
    env.random_obj_pos = True
    sample_size = 150000
    np_samples = np.zeros((sample_size, 256, 256))
    for i in tqdm(range(sample_size)):
        _ = env.reset()
        preprocessed_map = preprocess_images_np(env.cropped_prob_map)
        np_samples[i] = preprocessed_map

    np_to_h5(h5_save_path, np_samples)
    env.close()

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    rl_push('ViewPointEnvGUI-v0')
