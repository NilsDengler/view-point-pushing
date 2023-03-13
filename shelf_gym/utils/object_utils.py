import os
import pybullet as p
import numpy as np
import random
import time
SHELF_HEIGHT = 0.80
Z_TABLE_TOP = 0.85


class Objects():
    def __init__(self, sim, p_id):
        self.sim = sim
        self._p = p_id
        self.z_shelf_top = 0.905 #self.sim.default_z
        self.z_shelf_height = 0.905 #self.sim.default_z
        self.obj_ids = []
        self.obj_mass = []
        self.obj_names = []
        self.obj_positions = []
        self.obj_orientations = []
        self.target_id = None
        self.debug = False
        self.shelf_cover = self._p.loadURDF(self.sim.DEFAULT_PATH+'meshes/urdf/shelf_cover.urdf', [-2, -2, 0.1],
                                            self._p.getQuaternionFromEuler([0, 0, 0]), useFixedBase=True)
        self._p.changeDynamics(self.shelf_cover, -1, mass=-1)

    def create_temp_box(self, width, obj_init_pos):
        box_width = width
        box_height = 0.2
        box_z = self.z_shelf_top + (box_height/2)
        id1 = p.loadURDF(self.sim.DEFAULT_PATH + 'meshes/urdf/objects/slab1.urdf',
                         [obj_init_pos[0] - box_width /
                             2, obj_init_pos[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id2 = p.loadURDF(self.sim.DEFAULT_PATH + 'meshes/urdf/objects/slab1.urdf',
                         [obj_init_pos[0] + box_width /
                             2, obj_init_pos[1], box_z],
                         p.getQuaternionFromEuler([0, 0, 0]),
                         useFixedBase=True)
        id3 = p.loadURDF(self.sim.DEFAULT_PATH + 'meshes/urdf/objects/slab1.urdf',
                         [obj_init_pos[0], obj_init_pos[1] +
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        id4 = p.loadURDF(self.sim.DEFAULT_PATH + 'meshes/urdf/objects/slab1.urdf',
                         [obj_init_pos[0], obj_init_pos[1] -
                             box_width/2, box_z],
                         p.getQuaternionFromEuler([0, 0, np.pi*0.5]),
                         useFixedBase=True)
        return [id1, id2, id3, id4]

    def preload_all_objects(self, obj_info):
        obj_init_pos = [-0.8, 0]
        box_ids = self.create_temp_box(0.30, obj_init_pos)
        for path, mod_orn, mod_stiffness in obj_info:
            margin = 0.025
            r_x = random.uniform(
                obj_init_pos[0] - margin, obj_init_pos[0] + margin)
            r_y = random.uniform(
                obj_init_pos[1] - margin, obj_init_pos[1] + margin)
            yaw = random.uniform(0, np.pi)
            pos = [r_x, r_y, 1.0]

            obj_id, _, _ = self.load_obj(
                path, pos, yaw, mod_orn, mod_stiffness)
            for _ in range(10):
                self.sim.step_simulation(1)
            self.wait_until_still(obj_id, 150)

        self.wait_until_all_still()
        for handle in box_ids:
            p.removeBody(handle)
        self.update_obj_states()

    def load_obj(self, path, pos, yaw, mod_orn=False, mod_stiffness=False, name="default"):
        t0 = time.time()
        orn = self._p.getQuaternionFromEuler([0, 0, yaw])
        obj_id = self._p.loadURDF(path, pos, orn)
        # adjust position according to height
        aabb = self._p.getAABB(obj_id, -1)
        if mod_orn:
            minm, maxm = aabb[0][1], aabb[1][1]
            orn = self._p.getQuaternionFromEuler([0, np.pi * 0.5, yaw])
        else:
            minm, maxm = aabb[0][2], aabb[1][2]
        pos[2] += (maxm - minm) / 2
        #self._p.resetBasePositionAndOrientation(obj_id, pos, orn)
        # change dynamics
        mass = self._p.getDynamicsInfo(obj_id, -1)[0]
        if mod_stiffness:
            print("mod")
            self._p.changeDynamics(obj_id,
                                   -1, lateralFriction=1,
                                   rollingFriction=0.001,
                                   spinningFriction=0.002,
                                   restitution=0.01,
                                   contactStiffness=100000,
                                   contactDamping=0.0,
                                   mass=mass)
        else:
            #print("else")
            self._p.changeDynamics(obj_id,
                                   -1, lateralFriction=.5,
                                   rollingFriction=0.002,
                                   spinningFriction=0.001,
                                   restitution=0.01,
                                   mass=mass*100)
        r = random.uniform(0,1)
        g = random.uniform(0,0.5)
        b = random.uniform(0,1)

        self._p.changeVisualShape(obj_id, -1, rgbaColor=[r, g, b, 1], physicsClientId=self.sim.client_id)
        self.obj_mass.append(mass*20)
        self.obj_ids.append(obj_id)
        self.obj_names.append(name)
        self.obj_positions.append(pos)
        self.obj_orientations.append(orn)
        return obj_id, pos, orn

    def load_isolated_obj(self, path, x_pos, y_pos, z_pos, yaw, mod_orn=False, mod_stiffness=False):
        r_x = x_pos #random.uniform(x_pos - 0.1, x_pos + 0.1)
        r_y = y_pos #random.uniform(y_pos - 0.1, y_pos + 0.1)
        #yaw = np.pi#random.uniform(0, np.pi)

        pos = [r_x, r_y, z_pos]
        obj_id, _, _ = self.load_obj(path, pos, yaw, mod_orn, mod_stiffness)
        self.sim.step_simulation(self.sim.per_step_iterations)
        self.wait_until_still(obj_id)
        return obj_id
        #update_obj_states()

    def wait_until_still(self, objID, max_wait_epochs=10):
        for _ in range(max_wait_epochs):
            self.sim.step_simulation(self.sim.per_step_iterations)
            if self.is_still(objID):
                return

    def wait_until_all_still(self, max_wait_epochs=1000):
        for _ in range(max_wait_epochs):
            self.sim.step_simulation(1)
            if np.all(list(self.is_still(obj_id) for obj_id in self.obj_ids)):
                return
        if self.debug:
            print('Warning: Not still after MAX_WAIT_EPOCHS = %d.' %
                  max_wait_epochs)

    def is_still(self, handle):
        still_eps = 1e-2
        lin_vel, ang_vel = p.getBaseVelocity(handle)
        self._p.resetBaseVelocity(handle, [0,0,0],[0,0,0])
        return np.abs(lin_vel).sum() + np.abs(ang_vel).sum() < still_eps

    def update_obj_states(self):
        for i, obj_id in enumerate(self.obj_ids):
            pos, orn = self._p.getBasePositionAndOrientation(obj_id)
            self.obj_positions[i] = pos
            self.obj_orientations[i] = orn

    def remove_obj(self, obj_id):
        # Get index of obj in id list, then remove object from simulation
        idx = self.obj_ids.index(obj_id)
        self.obj_orientations.pop(idx)
        self.obj_positions.pop(idx)
        self.obj_ids.pop(idx)
        self._p.removeBody(obj_id)

    def remove_all_obj(self):
        self.obj_positions.clear()
        self.obj_orientations.clear()
        for obj_id in self.obj_ids:
            self._p.removeBody(obj_id)
        self.obj_ids.clear()

    def set_obj_mass(self, mass, exception=-1):
        for idx, o_id in enumerate(self.obj_ids):
            if o_id != exception:
                if mass != -1:
                    self._p.changeDynamics(o_id, -1, mass=self.obj_mass[idx])
                else:
                    self._p.changeDynamics(o_id, -1, mass=-1)
            #self.sim.step_simulation(self.sim.per_step_iterations)

    def reset_obj(self, obj_id, r_x, r_y, yaw=None):
        repeat = True
        #self.set_obj_mass(-1, obj_id)
        while repeat:
            if yaw is None:
                yaw = random.uniform(-np.pi/2, np.pi/2)
            pos = [r_x, r_y, self.z_shelf_height]
            orn = self._p.getQuaternionFromEuler([0, 0, yaw])

            aabb = self._p.getAABB(obj_id, -1)
            minm, maxm = aabb[0][2], aabb[1][2]

            pos[2] += (maxm - minm) / 2
            self._p.resetBasePositionAndOrientation(obj_id, pos, orn)
            # change dynamics
            self.wait_until_still(obj_id, 150)
            #self.wait_until_all_still(150)
            repeat = self.check_object_drop(obj_id, reset=True)
        #self.set_obj_mass(3, obj_id)
        self.update_obj_states()

    def check_object_drop(self, obj_id, reset=False):
        state = self._p.getBasePositionAndOrientation(obj_id)
        pos = state[0]
        orn = self._p.getEulerFromQuaternion(state[1])
        if reset:
            return orn[0] > 0.01 or orn[0] < -0.01 or orn[1] > 0.01 or orn[1] < -0.01 or pos[2] < 0.8 or pos[2] > 1.3
        return orn[0] > np.pi/4 or orn[0] < -np.pi/4 or orn[1] > np.pi/4 or orn[1] < -np.pi/4 or pos[2] < 0.8 or pos[2] > 1.3


    def check_all_object_drop(self, obj_ids):
        return any([self.check_object_drop(id) for id in obj_ids])


    def preload_objects(self, objects):
        path, mod_orn, mod_stiffness = objects.get_obj_info("CrackerBox")
        _ = self.load_isolated_obj(path, -0.0, 0.6, self.z_shelf_top,np.pi, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("CrackerBox")
        _ = self.load_isolated_obj(path, -1, 0.,  0.2, np.pi, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("TomatoSoupCan")
        _ = self.load_isolated_obj(path, -0.9, 0.2,  self.z_shelf_top, np.pi, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("TomatoSoupCan")
        _ = self.load_isolated_obj(path, -0.8, 0.15,  self.z_shelf_top, np.pi, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("TomatoSoupCan")
        _ = self.load_isolated_obj(path, -1, 0.15,  self.z_shelf_top, np.pi, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("ChipsCan")
        _ = self.load_isolated_obj(path, -0.7, 0.3,  self.z_shelf_top, np.pi, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("ChipsCan")
        _ = self.load_isolated_obj(path, -0.8, 0.3,  self.z_shelf_top, np.pi, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("ChipsCan")
        _ = self.load_isolated_obj(path, -0.9, 0.3,  self.z_shelf_top, np.pi, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("TomatoSoupCan")
        _ = self.load_isolated_obj(path, -0.8, 0.25,  self.z_shelf_top,np.pi,mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("CrackerBox")
        _ = self.load_isolated_obj(path,-0.9, 0.2,  self.z_shelf_top,np.pi, mod_orn, mod_stiffness)


    def load_object_scene_1(self, objects):
        #self.reset_obj()
        path, mod_orn, mod_stiffness = objects.get_obj_info("CrackerBox")
        _ = self.load_isolated_obj(path, -0.2, -0.8, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("TomatoSoupCan")
        _ = self.load_isolated_obj(path, 0.2, -0.8, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("TomatoSoupCan")
        _ = self.load_isolated_obj(path, 0.1, -0.8, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("TomatoSoupCan")
        _ = self.load_isolated_obj(path, 0.0, -0.8, mod_orn, mod_stiffness)
        path, mod_orn, mod_stiffness = objects.get_obj_info("PowerDrill")
        _ = self.load_isolated_obj(path, 0.1, -0.95, mod_orn, mod_stiffness)

    def set_object_on_target_line(self, o_id):
        pos, _ = self._p.getBasePositionAndOrientation(self.target_id)
        upper_bound_cam = self.sim.cam_world.get_target_pos_from_world_to_cam([0.0, self.sim.object_position_limit[1][0], 1.])
        lower_bound_cam = self.sim.cam_world.get_target_pos_from_world_to_cam([0.0, self.sim.object_position_limit[1][1], 1.])
        y_pos = random.uniform(lower_bound_cam[1], upper_bound_cam[1]) + 0.05
        world_pos = self.sim.cam_world.get_target_pos_from_cam_to_world([upper_bound_cam[0], y_pos, upper_bound_cam[2]])
        self._p.resetBasePositionAndOrientation(o_id, [pos[0], world_pos[1], pos[2]],
                                                self._p.getQuaternionFromEuler([0, 0, random.uniform(-np.pi/2, np.pi/2)]),
                                                physicsClientId=self.sim.client_id)
        self.wait_until_still(o_id)
        self.update_obj_states()
        return

    def place_target_relative(self, target_id, obj_id):
        pos, _ = self._p.getBasePositionAndOrientation(obj_id)
        pos = np.array(pos) + np.array(([0, 0.1, 0]))
        orn = self._p.getQuaternionFromEuler([0, 0, random.uniform(-np.pi/2, np.pi/2)])
        aabb = self._p.getAABB(target_id, -1)
        minm, maxm = aabb[0][2], aabb[1][2]
        pos[2] += (maxm - minm) / 2
        self._p.resetBasePositionAndOrientation(target_id, pos, orn)
        # change dynamics
        self.wait_until_still(target_id)
        self.update_obj_states()

    def reset_objects_fix(self, obj_config):
        current_obj_ids = []
        self.sim.initial_reset()
        for idx in range(len(obj_config)):
            self._p.resetBasePositionAndOrientation(obj_config[idx][0], obj_config[idx][1], obj_config[idx][2])
            current_obj_ids.append(obj_config[idx][0])
        return current_obj_ids


    def reset_objects_quarter(self, quarter):
        self.sim.initial_reset()
        self._p.resetBasePositionAndOrientation(self.shelf_cover, [0., 0.7, 1.1], self._p.getQuaternionFromEuler([0, 0, np.pi/ 2]))
        shuffled_ids = np.array(self.obj_ids).copy()
        np.random.shuffle(np.array(shuffled_ids).copy())
        shuffled_ids = shuffled_ids[:8]
        id_chunks = np.split(shuffled_ids, 4)
        for idx, q in enumerate(quarter):
            for id in id_chunks[idx]:
                self.reset_obj(id, q)
        self._p.resetBasePositionAndOrientation(self.shelf_cover, [3, 0.75, 0.9],self._p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        return shuffled_ids

    def reset_objects_for_rl(self, object_placing_bounds, sample_obj_num, sample_number=10):
        # put cover on shelf to prevent objects from falling out of the shelf

        time_sum = 0
        wrong_obj_spawn = True
        while wrong_obj_spawn:
            self.sim.initial_reset()
            self._p.resetBasePositionAndOrientation(self.shelf_cover, [0., 0.7, 1.1],
                                                    self._p.getQuaternionFromEuler([0, 0, np.pi / 2]))
            if sample_obj_num:
                obj_idxs = [self.obj_ids[0]]
                if len(self.obj_ids) > 1:
                    obj_idxs = np.array(self.obj_ids)[np.random.choice(len(self.obj_ids),
                                                                       size=sample_number,
                                                                       replace=False)]
                current_obj_ids = obj_idxs
            else:
                current_obj_ids = self.obj_ids
            r_x = np.random.uniform(object_placing_bounds[0][0], object_placing_bounds[0][1], len(current_obj_ids))
            r_y = np.random.uniform(object_placing_bounds[1][0], object_placing_bounds[1][1], len(current_obj_ids))
            t0 = time.time()
            if time_sum <= 5:
                for idx, id in enumerate(current_obj_ids):
                    self.reset_obj(id, r_x[idx], r_y[idx])
                for idx, id in enumerate(current_obj_ids):
                    repeat = self.check_object_drop(id, reset=True)
                    if repeat:
                        self.reset_obj(id, r_x[idx], r_y[idx])
                time_sum += time.time() - t0
                if len(current_obj_ids) > 0:
                    self.place_target_relative(self.sim.target_id, current_obj_ids[0])
                wrong_obj_spawn = self.check_all_object_drop(current_obj_ids)
            else:
                wrong_obj_spawn = False
                print("Spawning took too long, will skip this one")
        #remove cover from shelf
        self._p.resetBasePositionAndOrientation(self.shelf_cover, [3, 0.75, 0.9],
                                                self._p.getQuaternionFromEuler([0, 0, np.pi / 2]))
        self.update_obj_states()

        return current_obj_ids

class YcbObjects:
    def __init__(self, load_path, mod_orn=None, mod_stiffness=None, exclude=None):
        self.load_path = load_path
        self.mod_orn = mod_orn
        self.mod_stiffness = mod_stiffness
        with open(load_path + '/obj_list.txt') as f:
            lines = f.readlines()
            self.obj_names = [line.rstrip('\n') for line in lines]
        if exclude is not None:
            for obj_name in exclude:
                self.obj_names.remove(obj_name)

    def shuffle_objects(self):
        random.shuffle(self.obj_names)

    def get_obj_path(self, obj_name):
        if 'raw_model.urdf' in os.listdir(self.load_path + "/Ycb" + obj_name):
            return f'{self.load_path}/Ycb{obj_name}/raw_model.urdf'
        return f'{self.load_path}/Ycb{obj_name}/model.urdf'

    def check_mod_orn(self, obj_name):
        if self.mod_orn is not None and obj_name in self.mod_orn:
            return True
        return False

    def check_mod_stiffness(self, obj_name):
        if self.mod_stiffness is not None and obj_name in self.mod_stiffness:
            return True
        return False

    def get_obj_info(self, obj_name):
        return self.get_obj_path(obj_name), self.check_mod_orn(obj_name), self.check_mod_stiffness(obj_name)

    def get_n_first_obj_info(self, n):
        info = []
        for obj_name in self.obj_names[:n]:
            info.append(self.get_obj_info(obj_name))
        return info
