# Robotics Lab 2 — Forward Kinematics

## Objective

This project implements and validates forward kinematics for:

- HEAL 6-DoF robot
- FRANKA 7-DoF robot

The forward kinematics are calculated independently using Python and
validated using MuJoCo.

The same joint configuration q_bar is used in both Python and MuJoCo
for each robot.

---

## Robots

### HEAL — 6 DoF

Joint configuration:

```text
q_bar = [0.30, -0.40, 0.50, -0.20, 0.35, -0.25]
