import numpy as np
import math

def dist_normalized(c_min, c_max, dist):
    normed_min, normed_max = 0, 1
    x_normed = (dist - c_min) / (c_max - c_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return round(x_normed, 4)

def line_intersection(line1, line2):
    xdiff = (line1[0,0] - line1[1,0], line2[0,0] - line2[1,0])
    ydiff = (line1[0,1] - line1[1,1], line2[0,1] - line2[1,1])

    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    div = det(xdiff, ydiff)
    if div == 0:
        return None

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return np.array([x, y]).astype(np.int32)

def rotate(vector, angle, origin=(0,0)):
    angle = np.deg2rad(angle)
    ox, oy = origin
    px, py = vector

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return [qx, qy]

    #angle = np.deg2rad(angle)
    #R = np.array([[np.cos(angle), -np.sin(angle)],
    #              [np.sin(angle), np.cos(angle)]])
    #o = np.atleast_2d(origin)
    #p = np.atleast_2d(vector)
    #print(o, p)
    #return np.squeeze((R @ (p.T - o.T) + o.T).T)
    #theta = np.deg2rad(angle)
    #c, s = np.cos(theta), np.sin(theta)
    #R = np.array(((c, -s), (s, c)))
    #return R.dot(vector)


def get_angle_of_lines_with_common_point(center, p1, p2):
    d1 = p1 - center
    d2 = p2 - center
    angle1 = np.arctan2(d1[1], d1[0])
    angle2 = np.arctan2(d2[1], d2[0])
    return ((angle2 - angle1)*180)/np.pi

def translate_to_other_boundries(a,b,c,d,y):
    return (((y - a) * (c - d)) / (b - a)) + d

def angle_between(p1, p2):
    ang1 = np.arctan2(*p1[::-1])
    ang2 = np.arctan2(*p2[::-1])
    return np.rad2deg((ang1 - ang2) % (2 * np.pi))

def get_unit_vector(P, Q):
    if np.array_equal(P, Q): return np.zeros(2)
    PQ = Q - P
    #create unit vector
    uv = PQ / np.linalg.norm(PQ)
    ortho_vec = rotate(uv, 90)
    #print(uv)
    return ortho_vec

def isRotationMatrix(R) :
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

def rotationMatrixToEulerAngles(R):
    assert (isRotationMatrix(R))
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return np.array([x, y, z])


def quaternion_rotation_matrix(Q):
    # Extract the values from Q
    q0, q1, q2, q3 = Q[0], Q[1], Q[2], Q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = np.array([[r00, r01, r02],
                           [r10, r11, r12],
                           [r20, r21, r22]])

    return rot_matrix


def get_fk_ee(self):
    ls = self._p.getJointStates(self.sim.robot_id, range(1, 7))
    joint_angles = [i[0] for i in ls]
    pose_matrix = self.ur5_kin.forward(joint_angles, 'matrix')
    pos = pose_matrix[:3, 3:].reshape(3)
    pos[1] += 0.16
    pos[2] += 0.78
    euler = rotationMatrixToEulerAngles(pose_matrix[:3,:3])
    return np.array([pos,euler])

def get_ik_ee(self, ee_pos, hint_joints):
    ls = self._p.getJointStates(self.sim.robot_id, range(1, 7))
    hint_joints = [i[0] for i in ls]
    return self.ur5_kin.inverse(ee_pos, False, hint_joints)
    pos = ee_pos[0].reshape(3,1)
    pos[1] -= 0.16
    pos[2] -= 0.78
    orn = self._p.getQuaternionFromEuler(ee_pos[1])
    rot_mat = R.from_quat(orn).as_matrix()
    ee_mat = np.zeros((3, 4))
    ee_mat[:3, 3:] = pos
    ee_mat[:3,:3] = rot_mat
    return self.ur5_kin.inverse(ee_mat, True)[0]

def normalization(c_max, dist, c_min=0):
    normed_min, normed_max = 0, 1
    if (dist - c_min) == 0:
        return 0.
    if (c_max - c_min) == 0:
        print("IM IM WORST CASE: (c_max - c_min) == 0")
    x_normed = (dist - c_min) / (c_max - c_min)
    x_normed = x_normed * (normed_max - normed_min) + normed_min
    return round(x_normed, 4)