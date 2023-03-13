import time

import matplotlib.pyplot as plt
from stable_baselines3.common.env_util import make_vec_env
import gym, shelf_gym
import sys, os
import numpy as np
default_path = os.path.join(os.path.dirname(__file__), '../')
sys.path.append(default_path + 'network/')
sys.path.append(default_path + 'utils/')
sys.path.append(os.path.join(os.path.dirname(__file__), '../utils/LazyThetaStar/build/'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../../git/ur_ikfast/'))
sys.path.append(default_path + 'environment/')
sys.path.append(default_path)
import cv2
import open3d as o3d
from collections import deque
from math_utils import normalization, translate_to_other_boundries, angle_between
from rl_network_utils import preprocess_images_np, preprocess_images_torch
from scipy import ndimage
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
import imutils
from sklearn.cluster import MeanShift, estimate_bandwidth
def load_default_task():
    env = make_vec_env('MainEnvGUI-v0', n_envs=1)
    return env.envs[0].env.env

class ScanningTask():
    def __init__(self,env):
        self.env = env
        self.default_scanning_poses = np.array([[[0.0, 0.49, 0.850], [np.pi / 2, 0, -np.pi]],
                                                [[-0.441, 0.6, 0.850], [np.pi / 2, 0, 2.81]],
                                                [[-0.441, 0.66, 1.1], [(np.pi / 2) + 0.3, 0, 2.81]],
                                                [[0.441, 0.66, 1.1], [(np.pi / 2) + 0.3, 0, 3.48]],
                                                [[0.441, 0.66, 0.850], [(np.pi / 2) + 0.3, 0, 3.48]],
                                                [[0.0, 0.49, 0.850], [np.pi / 2, 0, -np.pi]]])
        self.setup_scanning()
        self.reset_scanning()

    def setup_scanning(self):
        _, _, _, _, first_hm, bounds, pixel_size, heightmap = self.generate_cam_data()
        self.initial_prob_map = self.env.cam_hand.draw_borders(np.ones_like(first_hm) * 0.5,  bounds, pixel_size)
        self.prob_map_history = deque(2*[self.initial_prob_map], 2)
        self.information_history = deque(np.zeros(2), 2)
        self.log_map = np.zeros_like(self.initial_prob_map)
        self.prob_map = self.initial_prob_map.copy()
        self.avg_height_map = np.zeros_like(self.initial_prob_map)
        self.binary_prob_map = self.initial_prob_map.copy()

    def reset_scanning(self):
        self.prob_map = self.initial_prob_map
        self.prob_map_history.appendleft(self.prob_map)
        self.prob_map_history.appendleft(self.prob_map)
        self.log_map = np.zeros_like(self.initial_prob_map)
        self.avg_height_map = np.zeros_like(self.initial_prob_map)
        self.binary_prob_map =  self.initial_prob_map.copy()

    def get_prob_map(self):
        self.scan_loop()
        return self.prob_map

    def scan_loop(self):
        cam_data = self.generate_cam_data()
        self.ortho_image = cam_data[4]
        self.mapping(cam_data)

    def generate_cam_data(self):
        rgb, depth, true_depth = self.env.cam_hand.get_hand_cam()
        self.rgb = rgb.copy()
        depth = self.env.cam_hand.remove_gripper(depth)
        self.pcd, pixel_size, bounds = self.env.cam_hand.get_pointcloud(depth)
        #pcd_shaped = self.pcd.copy().reshape(640*480, 3)
        #pcd = o3d.geometry.PointCloud()
        #pcd.points = o3d.utility.Vector3dVector(pcd_shaped)
        #o3d.visualization.draw_geometries([pcd])


        hm, fov = self.env.cam_hand.get_heightmap(self.pcd, rgb, bounds, pixel_size)
        # preproces hm data
        hm_dilate, hm_binarized = self.env.cam_hand.preprocess_heightmap(hm)
        return rgb, depth, true_depth, self.pcd, hm_binarized, bounds, pixel_size, hm_dilate

    def mapping(self, cam_data):
        # get indices of blocked, free and unknown cells
        self.blocked, self.free, self.unknown = self.env.cam_hand.get_indices(cam_data[4])
        # create prob map
        self.mapping_avg_height_map(cam_data[-1])
        self.mapping_binary_prob_map(self.blocked, self.free)
        self.mapping_log_odds(self.blocked, self.free)
        self.log_map = self.env.cam_hand.draw_borders(self.log_map, cam_data[5], cam_data[6])
        self.prob_map = self.env.cam_hand.draw_borders(self.prob_map, cam_data[5], cam_data[6])
        self.binary_prob_map = self.env.cam_hand.draw_borders(self.binary_prob_map, cam_data[5], cam_data[6])
        self.avg_height_map = self.env.cam_hand.draw_borders(self.avg_height_map, cam_data[5], cam_data[6], intensity=0.349)
        self.prob_map_history.appendleft(self.prob_map)

    def mapping_avg_height_map(self, heightmap):
        idx = np.logical_and(np.not_equal(self.avg_height_map, heightmap), (heightmap != 0))
        self.avg_height_map[idx] = heightmap[idx]

    def mapping_log_odds(self, blocked, free):
        self.log_map[free[:, 0], free[:, 1]] += -0.4
        self.log_map[blocked[:, 0], blocked[:, 1]] += 0.85
        self.log_map = np.clip(self.log_map, -2, 3.5)
        self.prob_map = 1-(1/(1 + np.exp(self.log_map.copy())))

    def mapping_binary_prob_map(self, blocked, free):
        current_binary_prob_map = np.ones_like(self.initial_prob_map)*0.5
        current_binary_prob_map[free[:, 0], free[:, 1]] = 0.
        current_binary_prob_map[blocked[:, 0], blocked[:, 1]] = 1.
        # fig, axs = plt.subplots(1, 2)
        # axs[0].imshow(current_binary_prob_map)
        # axs[1].imshow(self.binary_prob_map)
        # plt.show()
        idx = np.logical_and(np.not_equal(self.binary_prob_map, current_binary_prob_map), (current_binary_prob_map != 0.5))
        self.binary_prob_map[idx] = current_binary_prob_map[idx]

    def vp_entropy_calculation(self):
        roi_current = self.prob_map_history[0].copy()[73:, 31:286] #self.prob_map.copy()[73:, 31:286]
        roi_last = self.prob_map_history[1].copy()[73:, 31:286] #self.prob_map.copy()[73:, 31:286]
        roi_current[roi_last < 0.5] -= 1e-4
        roi_current[roi_last > 0.5] += 1e-4
        #roi_current = np.clip(roi_current, 0, 1)
        initial_unknown_num = roi_current.copy()[:151, 5:251].size
        current_unknown_num = np.count_nonzero(roi_current[:151, 5:251] == 0.5)
        return normalization(initial_unknown_num, current_unknown_num)


    def vp_information_gain_calculation(self, entropy_history, entropy_change_history):
        current_unknown_num, current_known_num, current_entropy = self.get_known_unknown_cells(self.prob_map_history[0].copy())
        last_unknown_num, last_known_num, last_entropy = self.get_known_unknown_cells(self.prob_map_history[1].copy())
        information_gain = normalization(last_unknown_num, (current_known_num - last_known_num))
        entropy_change = last_entropy - current_entropy
        entropy_change_history.appendleft(entropy_change)
        entropy_history.appendleft(current_entropy)
        return information_gain, entropy_history, entropy_change_history

    def get_known_unknown_cells(self, map):
        uk = np.count_nonzero(map[141:423, 54:425] == 0.5)
        k = np.count_nonzero(map[141:423, 54:425] != 0.5)
        e = round(uk / map[141:423, 54:425].size, 4)
        if k + uk != map[141:423, 54:425].size:
            print("Something went wrong: unknown and known number sum not up correctly")
        return uk, k ,e

    def compare_to_gt(self, map, gt):
        map_unknown_num, map_known_num, map_entropy = self.get_known_unknown_cells(map)
        gt_unknown_num, gt_known_num, gt_entropy = self.get_known_unknown_cells(gt)
        return normalization(gt_unknown_num, (map_known_num - gt_known_num)), map_entropy, gt_entropy

    def get_entropy_info(self):
        current_unknown_num, current_known_num = self.get_known_unknown_cells(self.prob_map_history[0].copy())
        initial_unknown_num = self.prob_map_history[0][141:423, 54:425].size
        entropy = normalization(initial_unknown_num, current_unknown_num)
        if current_known_num+current_unknown_num != initial_unknown_num:
            print("Something went wrong: unknown and known number sum not up correctly")
        return entropy, current_known_num, current_unknown_num, current_known_num

    def preprocess_prob_map(self, prob_map):
        crop_map = np.ones((300, 400))*0.5
        crop_map[5:296, 6:393] = prob_map[142:-1, 45:432]
        return crop_map

    def get_latent_space(self, prob_map, vae):
        #cut probmap to relevant area
        self.cropped_prob_map = self.preprocess_prob_map(prob_map)
        self.plt_image = np.flip(self.cropped_prob_map, axis=1)
        self.plt_image = np.rot90(self.plt_image, 2)
        reconst_img, latent_space, _, _, _ = vae(preprocess_images_torch(self.cropped_prob_map).to(self.env.torch_device))
        return list(latent_space.cpu().detach().numpy().reshape(32))

    def get_cnts_boxes_and_rectangles(self, img, size_thresh=10):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE )
        center_points = []
        boxes = []
        processed_cnts = []
        chulls = []
        max_area_idx = 0
        max_area = 0
        inner_count = 0
        for i, contour in enumerate(contours):
            area = cv2.contourArea(contour)
            if area >= size_thresh:
                if area > max_area:
                    max_area = area
                    max_area_idx = inner_count
                contour = self.add_offset(contour)
                chulls.append(cv2.convexHull(contour, False))
                processed_cnts.append(contour)
                rect = cv2.minAreaRect(contour)
                boxes.append(cv2.boxPoints(rect).astype(np.int32))
                #if area > 100:
                #    # find the center point of the contour
                M = cv2.moments(contour)
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                center_points.append((cX, cY))
                inner_count += 1
        return processed_cnts, chulls, boxes, center_points, max_area_idx

    def add_offset(self, c):
        c[:, :, 0] += 54
        c[:, :, 1] += 141
        return c


    def get_neighbors(self, img, cnts):
        binary = np.zeros((img.shape[0], img.shape[1]))
        for idx in range(len(cnts)):
            for c in cnts[idx]:
                #print(cnts[idx][:][:], cnts[idx][:][:][0], cnts[idx][:][:][1])
                binary[tuple(c[0][::-1])] = 1
        # Get offsets for row and column
        R_offset, C_offset = np.meshgrid(np.arange(-1, 2), np.arange(-1, 2))

        # Get row and column indices for places where elements are 1s
        R_match, C_match = np.nonzero(binary == 1)

        # Store number of matches as it would be frequently used
        N = R_match.size

        # Get offsetted row, col indices for all matches
        R_idx = (R_match[:, None, None] + R_offset).reshape(N, -1)
        C_idx = (C_match[:, None, None] + C_offset).reshape(N, -1)

        # Based on boundary conditions set invalid ones to zeros
        valid_mask = (R_idx >= 0) & (C_idx >= 0) & (R_idx < binary.shape[0]) & (C_idx < binary.shape[1])
        valid_mask[:, 4] = 0  # Set the pivot(self/center) ones to invalid

        # Using valid mask, "cut off" elements from each group of 9 elems
        cut_idx = valid_mask.sum(1).cumsum()

        # Finally form groups
        grps_R = np.split(R_idx[valid_mask], cut_idx)[:-1]
        grps_C = np.split(C_idx[valid_mask], cut_idx)[:-1]

        np.inter
        plt.imshow(binary)
        plt.show()

    def interpolate_cnts(self, cnts):
        #closethecontour, temporarily
        xc = [x[:], x(1)]
        yc = [y[:], y(1)]

        #currentspacingmaynot beequallyspaced
        dx = np.diff(xc)
        dy = np.diff(yc)

        # distancesbetweenconsecutivecoordiates
        dS = np.sqrt(np.power(dx,2) + np.power(dy,2))
        dS = [0, dS] # includingstartpoint

        # arc length, goingalong(around)snake
        d = np.cumsum(dS) # here is yourindependentvariable

    def draw_only_cont_points(self, img, cnts):
        #self.get_neighbors(img, cnts)

        i  = 0
        for idx in range(len(cnts)):
            c = np.array(cnts[idx]).reshape(-1, 2)
            c_sampled = c[::int(c.shape[0] * 0.1)]
            c_sorted = np.sort(c_sampled, axis=0)
            c_cut = c_sorted[:int(c_sorted.shape[0]/2)]
            for c in c_sampled:
                img = cv2.circle(img, tuple(c), 3, (255, 0, 0), -1)
            for c in c_cut:
                img = cv2.circle(img, tuple(c), 3, (0, 255, 0), -1)
                i+= 1
        return img

    def draw_boxes_on_image(self, img, cnts, chulls=None, boxes=None, cp=None, color=(0, 255, 0)):
        for idx in range(len(cnts)):
            img = cv2.drawContours(img, cnts[idx], -1, color, cv2.FILLED, cv2.LINE_8)
            if chulls: img = cv2.drawContours(img, chulls[idx], -1, color, cv2.FILLED, cv2.LINE_8)
            if boxes: img = self.draw_rec_on_image(img, boxes[idx], color)
            if cp:
                img = cv2.circle(img, cp[idx], 3, color, -1)
                img = cv2.putText(img, str(idx), cp[idx], cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        return img


    def draw_rec_on_image(self, img, points, color):
        img = cv2.line(img, tuple(points[0]), tuple(points[1]), color, 2)
        img = cv2.line(img, tuple(points[1]), tuple(points[2]), color, 2)
        img = cv2.line(img, tuple(points[2]), tuple(points[3]), color, 2)
        img = cv2.line(img, tuple(points[3]), tuple(points[0]), color, 2)
        return img

    def get_watershed(self, m):
        # Finding sure foreground area
        dist_transform = cv2.distanceTransform(m, cv2.DIST_L2, 3)
        plt.imshow(dist_transform)
        plt.show()

        ret, sure_fg = cv2.threshold(dist_transform, 0.6 * dist_transform.max(), 255, 0)
        sure_fg = np.uint8(sure_fg)
        ret, markers = cv2.connectedComponents(sure_fg)
        plt.imshow(markers)
        plt.show()
        markers = markers + 1
        img = cv2.merge((m, m, m))
        markers = cv2.watershed(img, markers)
        plt.imshow(markers)
        plt.show()
        img[markers == -1] = [255, 0, 0]
        return img


    def process_cluster(self):
        m = self.avg_height_map.copy()[141:423, 54:425]
        m[m < 0.07] = 0
        m = (m*255).astype(np.uint8)

        hist = np.histogram(m, bins=255, range=(1, 255))
        peaks = (hist[1][:-1][(hist[0] > 100)].astype(np.uint8))+1
        height_cluster_map = np.zeros((len(peaks), m.shape[0], m.shape[1])).astype(np.uint8)
        for idx, p in enumerate(peaks):
            height_cluster_map[idx][m == p] = 255
        return height_cluster_map

    def find_center_point(self, m, visualisation=False):
        #crop to relevant scene
        cm = m[141:423, 54:425]
        # get contours and polygon of objects in the scene
        cluster_map = self.process_cluster()
        c, h, b, cp = [], [], [], []
        for idx in range(cluster_map.shape[0]):
            c_t, h_t, b_t, cp_t, max_idx = self.get_cnts_boxes_and_rectangles(cluster_map[idx].reshape(cluster_map.shape[1], cluster_map.shape[2]), size_thresh=150)
            c.extend(c_t)
            h.extend(h_t)
            b.extend(b_t)
            cp.extend(cp_t)
        # get contours and polygon of unknown cells
        um = np.where(cm == 0.5, cm, 0) * 255
        um = cv2.normalize(um, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
        c_u, h_u, b_u, cp_u, max_idx_u = self.get_cnts_boxes_and_rectangles(um, size_thresh=160)
        object_cartesian_corners = self.obj_box_in_cartesian(b)
        unknown_area_center_points = self.unknown_cp_in_cartesian(cp_u)
        biggest_cp_u = unknown_area_center_points[max_idx_u]
        return_img = None
        if visualisation:
            #draw contours on image for better visualization
            return_img = cv2.merge((m, m, m))
            return_img = self.draw_only_cont_points(return_img, c)
            #return_img = self.draw_boxes_on_image(return_img, c, h, b, cp)
            #return_img = self.draw_boxes_on_image(return_img, c_u, h_u, cp=cp_u, color=(255, 0, 0))
        return return_img, object_cartesian_corners, unknown_area_center_points, biggest_cp_u

    def unknown_cp_in_cartesian(self, cp):
        transformed_cp = np.zeros((len(cp), 3))
        for idx, p in enumerate(cp):
            transformed_cp[idx] = self.env.cam_hand.otho_pixel_to_point(p)
        return transformed_cp

    def obj_box_in_cartesian(self, boxes):
        test = np.array(boxes)
        transformed_boxes = np.zeros((test.shape[0], test.shape[1], 3))
        for idx, b in enumerate(boxes):
            for jdx, p in enumerate(b):
                transformed_boxes[idx][jdx] = self.env.cam_hand.otho_pixel_to_point(p)
        return transformed_boxes

if __name__ == '__main__':
    environment = load_default_task()
    task = ScanningTask(environment)
    environment.obj.reset_obj(environment.obj.obj_ids[0], [[0, 0], [0.8, 0.8]], yaw=0)
    environment.obj.reset_obj(environment.target_id, [[0, 0], [0.95, 0.95]])
    environment.step_simulation(environment.per_step_iterations)
    while True:
        # reset environment
        #environment.initial_reset()
        # reset obstacles
        for i in range(len(environment.obj.obj_ids)):
            environment.obj.reset_obj(environment.obj.obj_ids[i], environment.target_position_limit)
        # reset Target
        environment.obj.reset_obj(environment.target_id, environment.target_position_limit)
        # step simulation
        environment.step_simulation(environment.per_step_iterations)
        # reset scanning
        task.reset_scanning()
        # while True:
        task.scanning()
        if task.target_detected:
            print("I saw the target during the scanning process at ", task.target_center_point)
            real_target_center = environment._p.getBasePositionAndOrientation(environment.target_id)[0]
            print("real center: ", real_target_center)
        else:
            print("I couldn't find the target during the scanning Process ")
        #plt.imshow(task.ortho_image)
        #plt.show()
        #environment.plt.plotImage([map])
