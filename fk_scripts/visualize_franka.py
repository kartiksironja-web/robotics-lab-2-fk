import mujoco
import mujoco.viewer
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

# Set the required joint configuration
for i in range(7):
    joint_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_JOINT,
        f"joint{i+1}"
    )

    qpos_id = model.jnt_qposadr[joint_id]
    data.qpos[qpos_id] = q_bar[i]

# Calculate robot pose
mujoco.mj_forward(model, data)

hand_id = mujoco.mj_name2id(
    model,
    mujoco.mjtObj.mjOBJ_BODY,
    "hand"
)

print("FRANKA q_bar:")
print(q_bar)

print("\nEnd-effector position:")
print(data.xpos[hand_id])

# Open viewer
with mujoco.viewer.launch_passive(model, data) as viewer:

    viewer.cam.lookat[:] = [0.0, 0.0, 0.9]
    viewer.cam.distance = 1.35
    viewer.cam.azimuth = 120
    viewer.cam.elevation = -15

    # Keep robot FIXED
    while viewer.is_running():
        viewer.sync()
