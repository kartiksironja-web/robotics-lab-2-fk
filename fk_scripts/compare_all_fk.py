import numpy as np
import mujoco

from heal_fk_python import forward_kinematics as heal_python_fk
from franka_fk_python import forward_kinematics as franka_python_fk


HEAL_XML = "robot_descriptions/single_arm_heal_effort_actuation_rs_mj.xml"
FRANKA_XML = "robot_descriptions/franka/panda.xml"


q_bar_heal = np.array([
    0.30,
   -0.40,
    0.50,
   -0.20,
    0.35,
   -0.25
])

q_bar_franka = np.array([
    0.20,
   -0.30,
    0.25,
   -1.00,
    0.40,
    1.00,
   -0.20
])


def get_mujoco_heal_position():
    model = mujoco.MjModel.from_xml_path(HEAL_XML)
    data = mujoco.MjData(model)

    for i in range(6):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"joint_{i+1}"
        )
        data.qpos[model.jnt_qposadr[joint_id]] = q_bar_heal[i]

    mujoco.mj_forward(model, data)

    site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        "right_center"
    )

    return data.site_xpos[site_id].copy()


def get_mujoco_franka_position():
    model = mujoco.MjModel.from_xml_path(FRANKA_XML)
    data = mujoco.MjData(model)

    for i in range(7):
        joint_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_JOINT,
            f"joint{i+1}"
        )
        data.qpos[model.jnt_qposadr[joint_id]] = q_bar_franka[i]

    mujoco.mj_forward(model, data)

    hand_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "hand"
    )

    return data.xpos[hand_id].copy()


# ---------------- HEAL ----------------

heal_python_position = heal_python_fk(q_bar_heal)[:3, 3]
heal_mujoco_position = get_mujoco_heal_position()

heal_difference = heal_python_position - heal_mujoco_position
heal_error = np.linalg.norm(heal_difference)


# ---------------- FRANKA ----------------

franka_python_position = franka_python_fk(q_bar_franka)[:3, 3]
franka_mujoco_position = get_mujoco_franka_position()

franka_difference = franka_python_position - franka_mujoco_position
franka_error = np.linalg.norm(franka_difference)


# ---------------- RESULTS ----------------

print("\n==========================================")
print("       ROBOTICS LAB 2 FK COMPARISON")
print("==========================================")

print("\nHEAL 6-DoF")
print("------------------------------------------")
print("Python position:")
print(heal_python_position)

print("MuJoCo position:")
print(heal_mujoco_position)

print("Difference:")
print(heal_difference)

print("Position error (m):")
print(heal_error)

print("Position error (mm):")
print(heal_error * 1000)


print("\nFRANKA 7-DoF")
print("------------------------------------------")
print("Python position:")
print(franka_python_position)

print("MuJoCo position:")
print(franka_mujoco_position)

print("Difference:")
print(franka_difference)

print("Position error (m):")
print(franka_error)

print("Position error (mm):")
print(franka_error * 1000)


print("\n==========================================")
print("FINAL VALIDATION")
print("==========================================")

if heal_error < 1e-4:
    print("HEAL: PASS")
else:
    print("HEAL: FAIL")

if franka_error < 1e-4:
    print("FRANKA: PASS")
else:
    print("FRANKA: FAIL")

if heal_error < 1e-4 and franka_error < 1e-4:
    print("\nALL FK RESULTS VALIDATED SUCCESSFULLY.")
else:
    print("\nCHECK FK IMPLEMENTATION.")
