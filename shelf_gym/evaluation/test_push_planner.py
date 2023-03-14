import sys, os
default_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(default_path + 'network/')
sys.path.append(default_path + 'utils/')
sys.path.append(default_path + 'utils/grid-map-raycasting')
sys.path.append(default_path + 'task/')
sys.path.append(default_path + 'environment/')
sys.path.append(default_path +'networks/vae/models/')
sys.path.append(default_path +'networks/push_prediction/')
sys.path.append(default_path)
#import for RL
from sb3_contrib import TQC
from stable_baselines3.common.env_util import make_vec_env
from tqdm import tqdm
import numpy as np
import gym, shelf_gym
import grid_map_raycasting as ray
from math_utils import get_angle_of_lines_with_common_point
import cv2
import matplotlib.pyplot as plt
import imutils
import torch
from push_model import push_predictions
from push_utils import load_checkpoint, get_loaders, check_accuracy
from scipy.spatial import distance


def run_random_vpp(env, obj_conf, max_steps, reset=True):
    if reset:
        _ = env.reset(evaluation=True, saved_obj_config=obj_conf)
    for idx in range(max_steps):
            action = env.action_space.sample()
            _, _, Done, _ = env.step(action)
    return env.prob_map,  env.scanning.avg_height_map, env.scanning.binary_prob_map.copy()

def run_trained_vpp(env, model, reset=True, obs=None ):
    if reset and obs is None:
        obs = env.reset()
    Done = False
    while not Done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, Done, info = env.step(action)
        # print('The agent collected a reward of %f during this iteration' % (reward))
    return env.prob_map,  env.scanning.avg_height_map, env.scanning.binary_prob_map.copy(), obs


def get_samples(map, height_map):
    final_push_samples = []
    final_direction_samples = []
    ray_mask = ray.rayCastGridMap(map)
    sample_indices = np.where(ray_mask == 1)
    push_samples = np.dstack((sample_indices[1], sample_indices[0]))[0, :]
    push_samples = push_samples[::10]
    directions = np.array([[20, 0], [0, 20], [20, 20], [-20, 0], [0, -20], [-20, -20], [-20, 20], [20, -20]])
    for c in push_samples:
        neighbors = directions + np.array(c)
        neighbor_values = height_map[neighbors[:, 1], neighbors[:, 0]]
        #print(neighbor_values)
        valid_neighbors = neighbors[neighbor_values > 0.07]
        #print(valid_neighbors)
        for n in valid_neighbors:
            final_push_samples.append(c)
            final_direction_samples.append(n)
    return final_push_samples, final_direction_samples


def preprocess_images_np(image):
    current_min, current_max = np.amin(image), np.amax(image)
    if current_min == current_max:
        return (image*0).astype(np.float16)
    normed_min, normed_max = 0, 1
    x_normed = (image - current_min) / (current_max - current_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return x_normed.astype(np.float16)


def get_push_images(map, push_points, directions):
    width, height = map.shape[:2]
    #push_points, directions = shuffle(push_points, directions, random_state=1)
    push_sample_batch = np.zeros((len(push_points), 400, 400))
    for idx in range(len(push_points)):
        dst = cv2.copyMakeBorder(map, int(width / 2), int(width / 2), int(height / 2), int(height / 2), cv2.BORDER_CONSTANT, None, value=0)
        new_width, new_height = dst.shape[:2]
        transformed_point = np.array(push_points[idx]) + np.array([height/2, width/2])
        ttx, tty = (new_height / 2) - (transformed_point[0]), (new_width / 2) - transformed_point[1]
        translated_image = imutils.translate(dst, ttx, tty)
        rot_angle = get_angle_of_lines_with_common_point(push_points[idx], np.array(push_points[idx]) + np.array([20, 0]), directions[idx])
        rotated_image = imutils.rotate(translated_image, rot_angle)
        cropped_image = rotated_image[int((new_width / 2) - 200):int((new_width / 2) + 200), int((new_height / 2) - 200):int((new_height / 2) + 200)]
        preprocessed_img = preprocess_images_np(cropped_image)
        push_sample_batch[idx] = preprocessed_img
    return push_sample_batch

def transform_push_img(map, push_point, direction):
    map = cv2.arrowedLine(map, (push_point[0], push_point[1]), (direction[0], direction[1]), 0.5, 4)
    width, height = map.shape[:2]
    dst = cv2.copyMakeBorder(map, int(width / 2), int(width / 2), int(height / 2), int(height / 2), cv2.BORDER_CONSTANT, None, value=0)
    new_width, new_height = dst.shape[:2]
    transformed_point = np.array(push_point) + np.array([height/2, width/2])
    ttx, tty = (new_height / 2) - (transformed_point[0]), (new_width / 2) - transformed_point[1]
    translated_image = imutils.translate(dst, ttx, tty)
    rot_angle = get_angle_of_lines_with_common_point(push_point, np.array(push_point) + np.array([20, 0]), direction)
    rotated_image = imutils.rotate(translated_image, rot_angle)
    result_img = rotated_image[int((new_width / 2) - 200):int((new_width / 2) + 200), int((new_height / 2) - 200):int((new_height / 2) + 200)]
    return result_img

def perform_best_push(env, position, direction):
    env.man.reset_robot(env.initial_parameters)
    pos = env.cam_hand.otho_pixel_to_point(position)
    angle = get_angle_of_lines_with_common_point(position, np.array(position) + np.array([20, 0]), direction)
    y_offset = np.random.uniform(0.0, 0.05)
    x_offset = np.random.uniform(-0.01, 0.01)
    target_pos = [pos[0]+x_offset, pos[1]+y_offset, 0.95, angle-90, 0.1]
    pushed_pose = env.man.perform_push_primitive(target_pos)
    env.man.reset_robot(env.initial_parameters)
    for i, obj_id in enumerate(list(env.current_obj_ids) + [env.shelf_id]):
        if obj_id != env.shelf_id:
            collision = env.check_for_tilted(obj_id, i)
    if collision.any():
        print("COLLISIION HAPPENED!!!!")
        return 1
    return 0

def show_best_push(map, push_point, direction):
    img = cv2.merge((map, map, map))
    img = cv2.arrowedLine(img, (push_point[0], push_point[1]), (direction[0], direction[1]), (255, 0, 0), 4)
    plt.imshow(img)
    plt.show()


def compare_obj_configs(env, conf_1, conf_2):
    e_dist = 0
    c_dist = 0
    for idx, c1 in enumerate(conf_1):
        #print(np.array(c1[0]), np.array(conf_2[idx][0]))
        e_dist += np.linalg.norm(np.array(c1[0]) - np.array(conf_2[idx][0]))
        ori_1 = env._p.getEulerFromQuaternion(c1[1])
        ori_2 = env._p.getEulerFromQuaternion(conf_2[idx][1])
        c_dist += distance.cosine(ori_1, ori_2)
    return e_dist + c_dist

def get_current_object_config(env, obj_config):
    current_object_configs = []
    for idx in range(len(obj_config)):
        pos, orn = env._p.getBasePositionAndOrientation(obj_config[idx])
        current_object_configs.append([pos, orn])
    return current_object_configs

def network_inference(model, data, device="cuda"):
    model.eval()
    final_prediction = np.zeros(0)
    for i in range(int(np.ceil(data.shape[0] / 32))):
        n, w, h = data[i*32:i*32+32].shape
        torch_data = torch.from_numpy(data[i*32:i*32+32].reshape(n, 1, w, h)).float().to(device)
        with torch.no_grad():
            x_pred_tag = torch.sigmoid(model(torch_data))
            predictions = x_pred_tag.cpu().detach().numpy().reshape(n)
        final_prediction = np.concatenate((final_prediction, predictions))
    return np.argmax(final_prediction)


def perform_pushing(env, model_vpp, model_pp, eval_num):
    for idx in tqdm(range(eval_num)):
        map, height_map, prob_map_1, obsv = run_trained_vpp(env, model_vpp)
        env.man.reset_robot(env.initial_parameters)
        Done = False
        step = 0
        current_diff = 0
        overall_displacement = 0
        while not Done:
            # get object config before push
            obj_conf_before = get_current_object_config(env, env.obj.obj_ids)

            # sample push candidates and predict best push
            push_sample, direction_samples = get_samples(map, height_map)
            push_sample_batch = get_push_images(height_map, push_sample, direction_samples)
            best_prediction = network_inference(model_pp, push_sample_batch)
            show_best_push(map, push_sample[best_prediction], direction_samples[best_prediction])
            # perform the best or random push
            collision = perform_best_push(env, push_sample[best_prediction], direction_samples[best_prediction])

            # get object after before push
            obj_conf_after = get_current_object_config(env, env.obj.obj_ids)
            overall_displacement += compare_obj_configs(env, obj_conf_before, obj_conf_after)
            # perform learned vpp again to see changes
            env.is_reset = True
            #perform first vpp sequence
            map, height_map, prob_map_2, obsv = run_trained_vpp(env, model_vpp, reset=False, obs=obsv)

            env.man.reset_robot(env.initial_parameters)
            ig, e_before, e_after = env.scanning.compare_to_gt(prob_map_1, prob_map_2)
            last_diff = current_diff
            current_diff = e_before - e_after

            if (current_diff - last_diff) < 0.01 or current_diff < 0.01 or collision > 0:
                Done = True
            step += 1

def set_up_prediction_network():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image_height = 400
    image_width = 400
    load_model_path = os.path.join(os.path.dirname(__file__), "../networks/push_prediction/saved_model/example_checkpoint.tar")
    model = push_predictions(image_width, image_height).to(device)
    load_checkpoint(torch.load(load_model_path), model)
    return model


def set_up_env(env_name, model_name):
    env = make_vec_env(env_name)
    saved_model_path = os.path.join(os.path.dirname(__file__), '../networks/rl/saved_models/' + model_name + "intermediate_saved_model")
    model = TQC.load(saved_model_path)
    return env.envs[0].env.env, model

def load_object_configs(load_path):
    return np.load(load_path, allow_pickle=True)

if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    env, model_vpp = set_up_env('ViewPointEnvGUI-v0', "TQC_example/model_1/")
    model_pp = set_up_prediction_network()
    perform_pushing(env, model_vpp, model_pp, 10)
