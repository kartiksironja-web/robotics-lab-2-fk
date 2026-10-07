import mujoco
import numpy as np

XML_PATH = "robot_descriptions/franka/panda.xml"

q_bar = np.array([
    0.20,
   -0.30,
    0.25,
   -1.00,
    0.40,
    1.00,
   -0.20
])

model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)

# Set the 7 joint values
for i in range(7):
    joint_name = f"joint{i + 1}"
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        joint_name
    )

    qpos_id = model.jnt_qposadr[joint_id]
    data.qpos[qpos_id] = q_bar[i]

# Compute forward kinematics
mujoco.mj_forward(model, data)

# FRANKA end-effector = hand body
hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

position = data.xpos[hand_id]

print("FRANKA MuJoCo Forward Kinematics")
print("--------------------------------")

print("\nq_bar (rad):")
print(q_bar)

print("\nEnd-effector position (m):")
print(position)

print("\nHand body ID:")
print(hand_id)
