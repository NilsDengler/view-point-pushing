"""
Code is based on
https://github.com/bulletphysics/bullet3/tree/master/examples/pybullet/gym/pybullet_envs
and
https://github.com/openai/gym/tree/master/gym/envs
"""

import pybullet as p
from pybullet_utils import bullet_client
import gym
from gym.utils import EzPickle
import pybullet_data

class BasePybulletEnv(gym.Env):
    def __init__(self, render=False, shared_memory=False, hz=240, use_egl=False):
        EzPickle.__init__(**locals())
        self._p = None
        self.step_size_fraction = 400#hz
        self._urdfRoot = pybullet_data.getDataPath()
        self.egl = use_egl
        self.shared_memory = shared_memory
        self.render = render
        self.pybullet_init()

    def _reset_base_simulation(self):
        self._p.resetSimulation()
        self._p.setGravity(0, 0, -9.81)

    def pybullet_init(self):
        render_option = p.DIRECT
        if self.render:
            render_option = p.GUI
        self._p = bullet_client.BulletClient(connection_mode=render_option)
        self._urdfRoot = pybullet_data.getDataPath()
        self._p.setAdditionalSearchPath(self._urdfRoot)

        self._egl_plugin = None
        # if not self.render:
        #     print("I will use the alternative renderer")
        #     assert sys.platform == 'linux', ('EGL rendering is only supported on ''Linux.')
        #     egl = pkgutil.get_loader('eglRenderer')
        #     if egl:
        #         self._egl_plugin = self._p.loadPlugin(egl.get_filename(), '_eglRendererPlugin')
        #     else:
        #         self._egl_plugin = self._p.loadPlugin('eglRendererPlugin')
        #     print('EGL renderering enabled.')

        self._p.configureDebugVisualizer(self._p.COV_ENABLE_GUI, 0)
        self._p.setPhysicsEngineParameter(enableFileCaching=0)
        self._p.setPhysicsEngineParameter(numSolverIterations=200)
        self._p.setTimeStep(1. / self.step_size_fraction)

        if self.render:
            self._p.resetDebugVisualizerCamera(
                cameraDistance=1.,
                cameraYaw=50.79996109008789,
                cameraPitch=-27.999988555908203,
                cameraTargetPosition=[0.05872843414545059, 0.4925108850002289, 0.9959999918937683],
            )

        self.client_id = self._p._client
        self._reset_base_simulation()
        control_frequency = 5
        self.per_step_iterations = int(self.step_size_fraction / control_frequency)
        self._p.configureDebugVisualizer(self._p.COV_ENABLE_RENDERING, 0)

    def step_simulation(self, num_steps):
        for _ in range(int(num_steps)):
            self._p.stepSimulation(physicsClientId=self.client_id)
            self._p.performCollisionDetection()

    def close(self):
        if self._egl_plugin is not None:
            p.unloadPlugin(self._egl_plugin)
        self._p.disconnect()


    def step(self):
        pass


    def reset(self):
        pass
