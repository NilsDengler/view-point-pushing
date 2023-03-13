#!/usr/bin/python3
from ur5_environment import RobotEnv
import gym
from camera_utils import Camera
from debug_utils import Debug, SphereMarker
from object_utils import YcbObjects, Objects
from plot_utils import Plot
from collision_utils import Collision
import sys, os
import numpy as np


class MainEnv(RobotEnv):
    def __init__(self, render=False, shared_memory=False, hz=240, use_egl=False):
        super().__init__(render=render, shared_memory=shared_memory, hz=hz, use_egl=use_egl)
        self.target_position_limit = ((-0.25, 0.25), (0.8, 1.))
        # place floor, shelf and table
        self.build_env()
        self.default_z = 0.97

        # initialize object class
        self.obj = Objects(self, self._p)
        self.ycb_objects = YcbObjects(os.path.join(os.path.dirname(__file__), '../meshes/urdf/ycb_objects'),
                                      mod_orn=['ChipsCan', 'MustardBottle', 'TomatoSoupCan'],
                                      mod_stiffness=['Strawberry'])
        self.obj.preload_objects(self.ycb_objects)
        #get example target obj
        path, mod_orn, mod_stiffness = self.ycb_objects.get_obj_info("TomatoSoupCan")
        self.target_id = self.obj.load_isolated_obj(path, -3, 0.25, 0.905, np.pi, mod_orn, mod_stiffness)

        #self.target_id = self._p.loadURDF(self.DEFAULT_PATH + 'meshes/urdf/target.urdf', [2., 2., 0.], self._p.getQuaternionFromEuler([0, 0, 0]))
        # initialize camera
        self.cam_hand = Camera(640, 480, self, self._p)

        # initalize collision checking class
        self.col = Collision(self, self._p)

        # initialize Debug values such as custom sliders to tune parameters (name of the parameter,range,initial value)
        self.deb = Debug(self, self._p)
        self.plt = Plot(self, self._p)

        #step simulation first time
        self.step_simulation(self.per_step_iterations)

        #save initial world state
        self.world_id = self._p.saveState()

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 1)
        self.global_steps = 0

    def _set_action_space(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

    def _set_observation_space(self):
        self.observation_space = gym.spaces.Box(np.array([0]), np.array([0]), dtype=np.float32)

    def build_env(self):
        self.planeID = self._p.loadURDF('plane.urdf')
        self.UR5Stand_id = self._p.loadURDF(self.DEFAULT_PATH+'meshes/urdf/objects/table.urdf', # urdf_change
                                            [0.225, -0.415, 0.45],
                                            self._p.getQuaternionFromEuler([0, 0, np.pi/2]), # urdf_change
                                            useFixedBase=True)
        self.shelf_id = self._p.loadURDF(self.DEFAULT_PATH+'meshes/urdf/shelf.urdf', [0., .9, 0.],
                                         self._p.getQuaternionFromEuler([0, 0, np.pi]),
                                         useFixedBase=True)
        self.wall_id = self._p.loadURDF(self.DEFAULT_PATH+'meshes/urdf/wall.urdf', [0., 1.13, 0.],
                                         self._p.getQuaternionFromEuler([0, 0, 0]),
                                         useFixedBase=True)

        #include objects to ompl planer
        #self.pb_ompl_interface.set_obstacles([self.UR5Stand_id, self.shelf_id, self.wall_id])


    def initial_reset(self):
        self._p.restoreState(self.world_id)
        self.man.reset_robot(self.initial_parameters)
        return
