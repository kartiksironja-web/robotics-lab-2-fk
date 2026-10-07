import mujoco
import numpy as np


XML_PATH = "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"


q_bar = np.array([
    0.30,
    -0.40,
    0.50,
    -0.20,
    0.35,
    -0.25
])


model = mujoco.MjModel.from_xml_path(XML_PATH)
data = mujoco.MjData(model)


# Set the six joint positions
for i in range(6):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"joint_{i + 1}"
    )

    qpos_id = model.jnt_qposadr[joint_id]

    data.qpos[qpos_id] = q_bar[i]


# Update all forward kinematics quantities
mujoco.mj_forward(model, data)


# Get the right_center site
site_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_SITE,
    "right_center"
)


position = data.site_xpos[site_id]


print("\nHEAL MuJoCo Forward Kinematics")
print("--------------------------------")

print("\nq_bar (rad):")
print(q_bar)

print("\nEnd-effector position (m):")
print(position)
