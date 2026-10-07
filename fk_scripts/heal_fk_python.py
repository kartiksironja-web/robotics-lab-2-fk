import numpy as np


def rot_x(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1, 0, 0],
        [0, c, -s],
        [0, s, c]
    ])


def rot_y(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, 0, s],
        [0, 1, 0],
        [-s, 0, c]
    ])


def rot_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0],
        [s,  c, 0],
        [0,  0, 1]
    ])


def make_transform(position, rotation):
    T = np.eye(4)
    T[:3, :3] = rotation
    T[:3, 3] = position
    return T


def joint_transform(axis, q):
    """
    Rotation caused by a revolute joint.
    """

    axis = np.array(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)

    x, y, z = axis

    K = np.array([
        [0, -z, y],
        [z, 0, -x],
        [-y, x, 0]
    ])

    R = (
        np.eye(3)
        + np.sin(q) * K
        + (1 - np.cos(q)) * (K @ K)
    )

    T = np.eye(4)
    T[:3, :3] = R

    return T


def forward_kinematics(q):
    """
    Forward kinematics for the 6-DoF HEAL robot.

    q = [q1, q2, q3, q4, q5, q6]

    The fixed transformations and joint axes are taken
    directly from the supplied MuJoCo XML.
    """

    q1, q2, q3, q4, q5, q6 = q

    T = np.eye(4)

    # --------------------------------------------------
    # link_1
    # pos = 0 0 0.171
    # joint_1 axis = 0 0 1
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0, 0.171],
        np.eye(3)
    )

    T = T @ joint_transform(
        [0, 0, 1],
        q1
    )

    # --------------------------------------------------
    # link_2
    # pos = 0 0.0875 0.1498
    # quat = 0.707105 0.707108 0 0
    #
    # This quaternion is approximately a 90-degree
    # rotation about X.
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0.0875, 0.1498],
        rot_x(np.pi / 2)
    )

    T = T @ joint_transform(
        [0, 0, -1],
        q2
    )

    # --------------------------------------------------
    # link_3
    # pos = 0 0.3 0
    # euler = 0 0 -1.57
    # joint_3 axis = 0 0 1
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0.3, 0],
        rot_z(-1.57)
    )

    T = T @ joint_transform(
        [0, 0, 1],
        q3
    )

    # --------------------------------------------------
    # link_4
    # pos = 0 0.1593 0.0875
    # euler = -1.57 0 0
    # joint_4 axis = 0 0 1
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0.1593, 0.0875],
        rot_x(-1.57)
    )

    T = T @ joint_transform(
        [0, 0, 1],
        q4
    )

    # --------------------------------------------------
    # link_5
    # pos = 0 0.03185 0.16105
    # euler = 0.50951 0 1.57
    # joint_5 axis = 0 0 1
    # --------------------------------------------------

    T = T @ make_transform(
        [0, 0.03185, 0.16105],
        rot_x(0.50951) @ rot_z(1.57)
    )

    T = T @ joint_transform(
        [0, 0, 1],
        q5
    )

    # --------------------------------------------------
    # end_effector
    # pos = 0 -0.1227 0.0654
    # quat = 0.707105 0.707108 0 0
    #
    # joint_6 axis = 0 0 -1
    # right_center is at [0,0,0]
    # --------------------------------------------------

    T = T @ make_transform(
        [0, -0.1227, 0.0654],
        rot_x(np.pi / 2)
    )

    T = T @ joint_transform(
        [0, 0, -1],
        q6
    )

    return T


if __name__ == "__main__":

    # Same q-bar that we will use in MuJoCo
    q_bar = np.array([
        0.30,
        -0.40,
        0.50,
        -0.20,
        0.35,
        -0.25
    ])

    T = forward_kinematics(q_bar)

    position = T[:3, 3]

    print("\nHEAL Forward Kinematics")
    print("-----------------------")

    print("\nq_bar (rad):")
    print(q_bar)

    print("\nEnd-effector transformation:")
    print(T)

    print("\nEnd-effector position (m):")
    print(position)
