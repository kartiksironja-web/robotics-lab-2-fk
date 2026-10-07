import numpy as np


def quat_to_rot(w, x, y, z):
    """
    Convert MuJoCo quaternion [w, x, y, z]
    to a 3x3 rotation matrix.
    """

    n = np.sqrt(w*w + x*x + y*y + z*z)

    w /= n
    x /= n
    y /= n
    z /= n

    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w),     2*(x*z + y*w)],
        [2*(x*y + z*w),     1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x*x + y*y)]
    ])


def make_transform(position, quaternion=None):

    T = np.eye(4)

    T[:3, 3] = position

    if quaternion is not None:
        T[:3, :3] = quat_to_rot(*quaternion)

    return T


def joint_rotation(q):
    """
    All seven Panda revolute joints use
    the default axis [0, 0, 1].
    """

    c = np.cos(q)
    s = np.sin(q)

    R = np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])

    T = np.eye(4)
    T[:3, :3] = R

    return T


def forward_kinematics(q):

    q1, q2, q3, q4, q5, q6, q7 = q

    T = np.eye(4)

    # --------------------------------------------------
    # link0
    # pos = -0.3 0 0.8
    # --------------------------------------------------

    T = T @ make_transform(
        [-0.3, 0, 0.8]
    )

    # --------------------------------------------------
    # link1
    # pos = 0 0 0.333
    # joint1
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0, 0.333]
    )

    T = T @ joint_rotation(q1)

    # --------------------------------------------------
    # link2
    # quat = 1 -1 0 0
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0, 0],
        [1, -1, 0, 0]
    )

    T = T @ joint_rotation(q2)

    # --------------------------------------------------
    # link3
    # pos = 0 -0.316 0
    # quat = 1 1 0 0
    # --------------------------------------------------

    T = T @ make_transform(
        [0, -0.316, 0],
        [1, 1, 0, 0]
    )

    T = T @ joint_rotation(q3)

    # --------------------------------------------------
    # link4
    # pos = 0.0825 0 0
    # quat = 1 1 0 0
    # --------------------------------------------------

    T = T @ make_transform(
        [0.0825, 0, 0],
        [1, 1, 0, 0]
    )

    T = T @ joint_rotation(q4)

    # --------------------------------------------------
    # link5
    # pos = -0.0825 0.384 0
    # quat = 1 -1 0 0
    # --------------------------------------------------

    T = T @ make_transform(
        [-0.0825, 0.384, 0],
        [1, -1, 0, 0]
    )

    T = T @ joint_rotation(q5)

    # --------------------------------------------------
    # link6
    # quat = 1 1 0 0
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0, 0],
        [1, 1, 0, 0]
    )

    T = T @ joint_rotation(q6)

    # --------------------------------------------------
    # link7
    # pos = 0.088 0 0
    # quat = 1 1 0 0
    # --------------------------------------------------

    T = T @ make_transform(
        [0.088, 0, 0],
        [1, 1, 0, 0]
    )

    T = T @ joint_rotation(q7)

    # --------------------------------------------------
    # hand
    # pos = 0 0 0.107
    # quat = 0.9238795 0 0 -0.3826834
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0, 0.107],
        [0.9238795, 0, 0, -0.3826834]
    )

    return T


if __name__ == "__main__":

    q_bar = np.array([
        0.20,
        -0.30,
        0.25,
        -1.00,
        0.40,
        1.00,
        -0.20
    ])

    T = forward_kinematics(q_bar)

    position = T[:3, 3]

    print("\nFRANKA Forward Kinematics")
    print("-------------------------")

    print("\nq_bar (rad):")
    print(q_bar)

    print("\nEnd-effector transformation:")
    print(T)

    print("\nEnd-effector position (m):")
    print(position)
