import numpy as np
import cv2

class Camera:
    def __init__(self, width, height, sim, p_id):
        self.sim = sim
        self._p = p_id
        self.width = width
        self.height = height
        self.near, self.far = 0.07, 1.
        self.fov = 58
        self.aspect = self.width / self.height
        self.intrinsic_matrix = None
        self.view_matrix = None
        self.projection_matrix = None
        self.focal_length = (float(self.width) / 2) / np.tan((np.pi * self.fov / 180) / 2)
        self.cx = float(self.width) / 2.
        self.pixel_size = -1
        self.bounds = np.asarray([[-0.5, 0.5], [0.5, 1.1], [0.7, 1.]])


    def get_hand_cam(self):
        # Center of mass position and orientation (of link-7)
        self.intrinsic_matrix = np.array([[self.focal_length, 0, float(self.width) / 2],
                                          [0, self.focal_length, float(self.width) / 2],
                                          [0, 0, 1]])
        self.projection_matrix = self._p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        com_p, com_o, _, _, _, _ = self._p.getLinkState(self.sim.robot_id, self.sim.camera_link, computeForwardKinematics=True)
        rot_matrix = self._p.getMatrixFromQuaternion(com_o)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)
        # Initial vectors
        init_camera_vector = (0, 0, 1)  # z-axis
        init_up_vector = (0, 1, 0)  # y-axis
        # Rotated vectors
        camera_vector = rot_matrix.dot(init_camera_vector)
        up_vector = rot_matrix.dot(init_up_vector)
        self.view_matrix = self._p.computeViewMatrix(com_p, com_p + 0.1 * camera_vector, up_vector)
        return self.get_image()

    def get_image(self):
        images = self._p.getCameraImage(self.width, self.height, self.view_matrix, self.projection_matrix,
                                          shadow=False, renderer=self._p.ER_BULLET_HARDWARE_OPENGL)
        rgb = np.array(images[2])[:,:,:3]
        depth = images[3]
        self.segmented = images[4]
        true_depth = self.far * self.near / (self.far - (self.far - self.near) * depth)
        return rgb, depth, true_depth

    def define_world_cam(self):
        cyaw = 181.98989868164062
        cpitch = -35.599998474121094
        croll = 0
        cdist = 1.0
        target_pos = [0.013031159527599812, -0.9652420878410339, 0.8137348294258118]
        self.view_matrix = np.asarray(self._p.computeViewMatrixFromYawPitchRoll(target_pos, cdist, cyaw,
                                                                                cpitch, croll, 2))
        self.projection_matrix = self._p.computeProjectionMatrixFOV(self.fov, self.aspect, self.near, self.far)
        self.intrinsic_matrix = np.array([[self.focal_length, 0, float(self.width) / 2],
                                     [0, self.focal_length, float(self.width) / 2],
                                     [0, 0, 1]])


    def depth_to_point(self, depth):
        # based on https://stackoverflow.com/questions/59128880/getting-world-coordinates-from-opengl-depth-buffer
        # create a 4x4 transform matrix that goes from pixel coordinates (and depth values) to world coordinates
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        tran_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
        # create a grid with pixel coordinates and depth values
        y, x = np.mgrid[-1:1:2 / self.height, -1:1:2 / self.width]
        y *= -1.
        x, y, z = x.reshape(-1), y.reshape(-1), depth.reshape(-1)
        h = np.ones_like(z)

        pixels = np.stack([x, y, z, h], axis=1)
        # filter out "infinite" depths
        #pixels = pixels[z < 0.99]
        pixels[:, 2] = 2 * pixels[:, 2] - 1

        # turn pixels to world coordinates
        points = np.matmul(tran_pix_world, pixels.T).T
        points /= points[:, 3: 4]
        points = points[:, :3]

        return points


    def get_pointcloud(self, depth):
        pc = self.depth_to_point(depth)
        # get pixel size for further calculations
        if self.pixel_size == -1:
            self.pixel_size = [0.002082267481217665, 0.001381592622035263]
            #self.pixel_size = [(max(pc[:, 0]) - min(pc[:, 0])) / self.width,
            #                   (max(pc[:, 1]) - min(pc[:, 1])) / self.height]
        bounds = np.asarray([[-0.5, 0.5], [0.5, 1.1], [0.85, 1.2]])
        pc = np.array(pc).reshape(self.width,  self.height, 3)
        return pc, self.pixel_size, bounds


    def get_heightmap(self, points, colors, bounds, pixel_size):
        """Get top-down (z-axis) orthographic heightmap image from 3D pointcloud.

        Args:
          points: HxWx3 float array of 3D points in world coordinates.
          colors: HxWx3 uint8 array of values in range 0-255 aligned with points.
          bounds: 3x2 float array of values (rows: X,Y,Z; columns: min,max) defining
            region in 3D space to generate heightmap in world coordinates.
          pixel_size: float defining size of each pixel in meters.

        Returns:
          heightmap: HxW float array of height (from lower z-bound) in meters.
          colormap: HxWx3 uint8 array of backprojected color aligned with heightmap.
        """
        width = int(np.round((bounds[0, 1] - bounds[0, 0]) / (pixel_size[0])))
        height = int(np.round((bounds[1, 1] - bounds[1, 0]) / (pixel_size[1])))
        heightmap = np.zeros((height, width), dtype=np.float32)
        fov = []
        #colormap = np.ones((height, width, colors.shape[-1]), dtype=np.uint8)*100
        if points.size != 0:
            # Filter out 3D points that are outside of the predefined bounds.
            ix = (points[Ellipsis, 0] >= bounds[0, 0]) & (points[Ellipsis, 0] < bounds[0, 1])
            iy = (points[Ellipsis, 1] >= bounds[1, 0]) & (points[Ellipsis, 1] < bounds[1, 1])
            iz = (points[Ellipsis, 2] >= bounds[2, 0]) & (points[Ellipsis, 2] < bounds[2, 1])
            valid = ix & iy & iz
            points = points[valid]
            #colors = colors[valid]
            # Sort 3D points by z-value, which works with array assignment to simulate
            # z-buffering for rendering the heightmap image.
            iz = np.argsort(points[:, -1])
            points = points[iz]
            px = np.int32(np.floor((points[:, 0] - bounds[0, 0]) / pixel_size[0]))
            py = np.int32(np.floor((points[:, 1] - bounds[1, 0]) / pixel_size[1]))
            px = np.clip(px, 0, width - 1)
            py = np.clip(py, 0, height - 1)
            heightmap[py, px] = points[:, 2] - bounds[2, 0]
            fov.append([py, px])
        return heightmap, fov

    def draw_borders(self, heightmap, bounds, pixel_size, intensity=1.):
        heightmap = self.draw_shelf_borders_hm([0.3925, 0.7, 0.99], [0.3925, 1.089, 0.99], heightmap, bounds, pixel_size, intensity)
        heightmap = self.draw_shelf_borders_hm([-0.3975, 0.7, 0.99], [-0.3975, 1.089, 0.99], heightmap, bounds, pixel_size, intensity)
        heightmap = self.draw_shelf_borders_hm([0.3925, 1.089, 0.99], [-0.3975, 1.089, 0.99], heightmap, bounds, pixel_size, intensity)
        return heightmap

    def remove_gripper(self, depth, open=False):
        if open:
            depth[193:, :53] = 1.
            depth[193:, 199:] = 1.
        else:
            depth[self.segmented == 0] = 1. #0.99
        self.without_gripper = depth
        return depth

    def preprocess_heightmap(self, hm):
        preprocessed_hm = cv2.morphologyEx(hm, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
        hm_binarized = preprocessed_hm.copy()
        hm_binarized[hm_binarized >= 0.075] = 255 # blocked
        hm_binarized[hm_binarized == 0.0] = 100 # unknown
        hm_binarized[hm_binarized <= 0.074] = 0 # free
        return preprocessed_hm, hm_binarized

    def get_indices(self, map, block_val=255, free_val=0, unkn_val=100):
        blocked = np.where(map == block_val)
        free = np.where(map == free_val)
        unknown = np.where(map == unkn_val)
        return np.stack((blocked[0], blocked[1]), axis=-1), \
               np.stack((free[0], free[1]), axis=-1), \
               np.stack((unknown[0], unknown[1]), axis=-1)


    def otho_pixel_to_point(self, pix):
        point = np.zeros(3)
        point[0] = (pix[0] * self.pixel_size[0]) + self.bounds[0, 0]
        point[1] = (pix[1] * self.pixel_size[1]) + self.bounds[1, 0]
        point[2] = self.sim.default_z
        return point

    def point_to_otho_pixel(self, point):
        px = np.int32(np.floor((point[0] - self.bounds[0, 0]) / self.pixel_size[0]))
        py = np.int32(np.floor((point[1] - self.bounds[1, 0]) / self.pixel_size[1]))
        return [px, py]

    def check_if_pix_in_bounds(self, pix, w, h):
        if pix[0] < 0:
            pix[0] = 0
        if pix[0] > w-1:
            pix[0] = w-1
        if pix[1] < 0:
            pix[1] = 0
        if pix[1] > h-1:
            pix[1] = h-1
        return np.array(pix)

    def draw_shelf_borders_hm(self, point_1, point_2, img, bounds, pixel_size, intensity):
        px_1 = np.int32(np.floor((point_1[0] - bounds[0, 0]) / pixel_size[0]))
        py_1 = np.int32(np.floor((point_1[1] - bounds[1, 0]) / pixel_size[1]))
        px_2 = np.int32(np.floor((point_2[0] - bounds[0, 0]) / pixel_size[0]))
        py_2 = np.int32(np.floor((point_2[1] - bounds[1, 0]) / pixel_size[1]))
        return cv2.line(img, (px_1, py_1), (px_2, py_2), intensity, 5)

    def get_target_pos_from_world_to_cam(self, point):
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        trans_cam = np.matmul(proj_matrix, view_matrix)
        ps_homogeneous = np.append(point, 1.)
        ps_transformed = np.matmul(trans_cam, ps_homogeneous.T).T
        ps_transformed /= ps_transformed[-1]
        return ps_transformed[:3]

    def get_cam_pos_in_world(self):
        cam_pose = np.linalg.inv(np.array(self.view_matrix).reshape(4, 4).T)
        cam_pose[:, 1:3] = -cam_pose[:, 1:3]
        return cam_pose

    def get_target_pos_from_cam_to_world(self, cam_point):
        proj_matrix = np.asarray(self.projection_matrix).reshape([4, 4], order="F")
        view_matrix = np.asarray(self.view_matrix).reshape([4, 4], order="F")
        trans_pix_world = np.linalg.inv(np.matmul(proj_matrix, view_matrix))
        # turn pixels to world coordinates
        cam_point_homogeneous = np.append(cam_point, 1.)
        points = np.matmul(trans_pix_world, cam_point_homogeneous.T).T
        points /= points[-1]
        return points[:3]