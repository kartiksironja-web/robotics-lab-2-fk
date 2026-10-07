import mujoco
import mujoco.viewer
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

# Set the required joint configuration
for i in range(6):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"joint_{i+1}"
    )

    qpos_id = model.jnt_qposadr[joint_id]
    data.qpos[qpos_id] = q_bar[i]

# Calculate the robot pose
mujoco.mj_forward(model, data)

print("HEAL q_bar:")
print(q_bar)

print("\nEnd-effector position:")
print(data.xpos[mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "end_effector"
)])

# Open viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    # Camera
    viewer.cam.lookat[:] = [0.0, 0.1, 0.4]
    viewer.cam.distance = 1.6
    viewer.cam.azimuth = 90
    viewer.cam.elevation = -10

    # Keep robot FIXED
    while viewer.is_running():
        viewer.sync()

