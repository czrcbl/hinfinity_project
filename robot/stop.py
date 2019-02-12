from serial_com import RobotCom
import numpy as np


try:
    com = RobotCom()
    print('Port Openned:')
except Exception as e:
    print(e)

com.send_control_signal(np.array([0, 0, 0]))
