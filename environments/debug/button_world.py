"""
Simple environment with a push button. Made for debug.
"""
from __future__ import division, absolute_import, print_function

import os
import math
from datetime import datetime

import pybullet_data
import pybullet as p

p.connect(p.GUI)
urdf_root_path = pybullet_data.getDataPath()
p.loadURDF(os.path.join(urdf_root_path, "plane.urdf"), [0, 0, -0.3], useFixedBase=True)

button_uid = p.loadURDF("/urdf/simple_button.urdf", [0, 0, 0])
glider_idx = 1
button_link_idx = 1

p.setGravity(0, 0, -10)

t = 0
useSimulation = True
useRealTimeSimulation = False
p.setRealTimeSimulation(useRealTimeSimulation)

button_pos_slider = p.addUserDebugParameter("button_pos", 0, 1, 0.1)


while True:
    if useRealTimeSimulation:
        dt = datetime.now()
        t = (dt.second / 60.) * 2. * math.pi
    else:
        t = t + 0.1

    if useSimulation and not useRealTimeSimulation:
        p.stepSimulation()

    button_position = p.readUserDebugParameter(button_pos_slider)
    # p.applyExternalForce(button_uid, button_link_idx, (0, 0, 10), (0, 0, 0), p.LINK_FRAME)
    p.setJointMotorControl2(button_uid, glider_idx, controlMode=p.POSITION_CONTROL, targetPosition=button_position)
