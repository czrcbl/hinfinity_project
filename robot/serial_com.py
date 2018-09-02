import serial
import numpy as np
import time


class RecvData:
    
    def __init__(self, **kargs):
        self.__dict__.update(kargs)
        
    def __str__(self):
        s = ''
        for key, val in self.__dict__.items():
            s += f'{key}: {val}\n'
            
        return s[:-1]
    
    def __repr__(self):
        s = 'RecvData('
        for key, val in self.__dict__.items():
            s = s + key + '=' + str(val) + ', '
    
        return s[:-2] + ')'
        


class RobotCom:

    def __init__(self, port='/dev/ttyUSB0', baudrate=115200,
                 address="13A20040B09872"):
        self.serial = serial.Serial(port)
        self.serial.baudrate = baudrate
        self.address = address

    def control_to_parameter(self, control_vector):
        """
        Converts the control signal input to a bytearray containing the PWM
        values and the rotation directions that can be directly used on the
        message.
        """
        parameter = bytearray(4)
        num_map = (1, 2, 4)
        # new_control = 255 - (255/6.0) * (6 - control_vector)
        new_control = 255 / 6.0 * control_vector

        for i in range(len(new_control)):
            u = new_control[i]
            if u < 0:
                u = -1 * u
                parameter[3] += num_map[i]
            if np.abs(u) > 255:
                u = 255
            # TODO: why this & 0xFF
            parameter[i] = int(u) & 0xFF
            # parameter[i] = int(u)

        return parameter

    def make_message(self, command, parameter):

        address = self.address
        if len(address) < 16:
            address = address.zfill(16)

        buff = bytearray(18)
        buff[0] = 0x7E
        buff[1] = 0x00
        buff[3] = 0x10
        buff[4] = 0x00

        for i in range(0, 16, 2):
            buff[5 + int(i / 2)] = int(address[i:i + 2], 16)

        buff[13] = 0xFF
        buff[14] = 0xFE
        buff[15] = 0x00

        buff[16] = 0x00
        buff[17] = command & 0xFF

        buff += parameter

        sum_arr = bytearray(1)
        sum_arr[0] = self.check_sum(buff[3:])
        buff += sum_arr

        buff[2] = len(str(parameter)) + 15

        return buff

    def send_command(self, message):
        self.serial.write(message)

    def check_sum(self, data):
        sum_ = 0
        for n in data:
            sum_ += n

        return 0xFF - (sum_ & 0x00FF)

    def receive_message(self):
        # TODO: check the exact number of bytes, 47 is a magic number
        n_bytes = 47
        try:
            mess = self.serial.read(n_bytes)
        except Exception as e:
            print(e)

        # Flush the remaining bytes, so next read will start from the beginning
        # of next message
        self.serial.flushInput()
        data = self.parse_message(mess)

        return data

    def receive_message2(self):
        """For testing purposes"""
        n_bytes = 47
        # TODO: check this /\
        try:
            mess = self.serial.read(n_bytes)
        except Exception as e:
            print(e)

        # Flush the remaining bytes, so next read will start from the beginning
        # of next message
        self.serial.flushInput()
        data = self.parse_message(mess)

        return data, mess

    def parse_message(self, message):
        """Parse the raw message received from the robot"""

        data = RecvData()
        buff = bytearray(message)

        # The vel is on rpm * 10
        vel = (buff[16] * 256 + buff[17]) / 600.0 * 2 * np.pi
        if (0x01 & buff[36]) == 1:
            data.m1_vel = vel
        else:
            data.m1_vel = -vel
        # data.m1_vel = (-1)**int(not (0x01 & buff[36])) * vel
        vel = (buff[18] * 256 + buff[19]) / 600.0 * 2 * np.pi
        if (0x02 & buff[36]) == 2:
            data.m2_vel = vel
        else:
            data.m2_vel = -vel
        # data.m2_vel = (-1)**int(not (0x02 & buff[36])) * vel
        vel = (buff[20] * 256 + buff[21]) / 600.0 * 2 * np.pi
        if (0x04 & buff[36]) == 4:
            data.m3_vel = vel
        else:
            data.m3_vel = -vel
        # data.m3_vel = (-1)**int(not (0x04 & buff[36])) * vel

        # TODO: Check the scale of the values, and fix it
        # TODO: Extract the other parameters from the buffer
        data.m1_curr = buff[22] * 256 + buff[23]
        data.m2_curr = buff[24] * 256 + buff[25]
        data.m3_curr = buff[26] * 256 + buff[27]
        data.x_acell = buff[28] * 256 + buff[29]
        data.y_acell = buff[30] * 256 + buff[31]
        data.ang_vel = buff[32] * 256 + buff[33]
        data.compass = buff[34] * 256 + buff[35]
        data.m1_dutycycle = buff[37]
        data.m2_dutycycle = buff[38]
        data.m3_dutycycle = buff[39]

        return data

    def send_control_signal(self, control_signal):
        """Control signal is a 3-element numpy array"""
        parameter = self.control_to_parameter(control_signal)
        message = self.make_message(2, parameter)
        self.send_command(message)

    def init_serial(self):
        """The robot starts to send the data once it receives this message."""
        parameter = ''
        message = self.make_message(1, parameter)
        self.send_command(message)

    def stop_motors(self):
        control_signal = np.array([0, 0, 0])
        self.send_control_signal(control_signal)


if __name__ == '__main__':

    pass

    # Test: Ramp
    # init_serial(ser)
    # control_signal = np.array([0, 0, 0])
    # for n in range(50):
    #     data = recv_message(ser)
    #     print(data.m1_vel)
    #     print(data.m2_vel)
    #     print(data.m3_vel)
    #     control_signal = control_signal + 0.01
    #     send_control_signal(control_signal, ser)
    #
    # for n in range(50):
    #     data = recv_message(ser)
    #     print(data.m1_vel)
    #     print(data.m2_vel)
    #     print(data.m3_vel)
    #     control_signal = control_signal - 0.01
    #     send_control_signal(control_signal, ser)

    # parameter = control_to_parameter(np.array([0, 0, 0]))
    # # Command 2 sets the pwm values for the motors
    # print(parameter)
    #
    # message = make_message("13A20040B09872" , 2, parameter)
    # print(message)
    # send_command(message, ser)

    # Exp1: Medicoes zero
    # init_serial(ser)
    # control_signal = np.array([0, 2, -2])
    # for n in range(4):
    #     send_control_signal(control_signal, ser)
    #     data = recv_message(ser)
    #     vel = np.array([data.m1_vel, data.m2_vel, data.m3_vel])
    #     print(vel)

    # init_serial(ser)
    # for i in range(10):
    #     ser.read(1)
    #     ser.flushInput()

    # from main import wheel2states
    # control_signal = np.array([0, 2, -2])
    # for n in range(8):
    #     data = recv_message(ser)
    #     vel = np.array([data.m1_vel, data.m2_vel, data.m3_vel])
    #     print(vel)
    #     states = wheel2states(vel)
    #     print(states)
    #     send_control_signal(control_signal, ser)
    #
    # send_control_signal(np.array([0, 0, 0]), ser)

    # init_serial(ser)
    # control_signal = np.array([0, 0.2, -0.2])
    # for n in range(8):
    #     data = recv_message(ser)
    #     vel = np.array([data.m1_vel, data.m2_vel, data.m3_vel])
    #     print(vel)
    #     control_signal = control_signal + np.array([0, 0.2, -0.2])
    #     send_control_signal(control_signal, ser)
    #
    # send_control_signal(np.array([0, 0, 0]), ser)
