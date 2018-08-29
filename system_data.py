import numpy as np
import control
from scipy.io import savemat

class Nldat:
    def __init__(self, **kargs):
        self.__dict__.update(kargs)
    
I = np.eye(3)
Z = np.zeros((3, 3))

Bv = 0.7
Bvn = 0.7
Bw = 0.011
 
Cv = 0.28
Cvn = 0.14
Cw = 0.0086

Fsv = 0.80
Fsvn = 0.65
Fsw = 0.02

M = 1.551
J = 0.0062

delta = np.pi/6

b = 0.1
r = 0.0505
l = 19
 
La = 0.00011
Ra = 1.69
Kv = 0.0059
Kt = 0.0059

actuator_saturation = 6

Ae = -3*l*l*Kt*Kv/(r*r*Ra)
Be = l*Kt/(r*Ra)

Ac = np.diag([(Ae/2-Bv)/M, (Ae/2-Bvn)/M, (Ae*b*b - Bw)/J])
Bc = np.array([[0, np.sqrt(3)/(2*M), -np.sqrt(3)/(2*M)], 
               [-1/M, 1/(2*M), 1/(2*M)], 
               [ b/J, b/J, b/J] ])*Be;
Cc = I
Dc = Z
Kc = np.diag([-Cv/M, -Cvn/M, -Cw/J])
Kf = np.diag([(-Fsv + Cv)/M, (-Fsvn + Cvn)/M, (-Fsw + Cw)/J])

nldat = Nldat(Kc=Kc, Kf=Kf)

Ts = 0.06

csys = control.ss(Ac, Bc, Cc, Dc)
dsys = control.c2d(csys, Ts, 'zoh')

[Ad, Bd, Cd, Dd] = control.ssdata(dsys)

#converts the wheel velocities into the states
W2S = np.array([[0, np.sqrt(3) * r / 3.0, -np.sqrt(3) * r / 3.0],
               [-2.0 * r / 3.0, r / 3.0, r / 3.0],
               [r / (3.0 * b), r / (3.0 * b), r / (3.0 * b)]])

#converts the states into the wheel velocities
S2W = np.array([[0, -1, b],
              [np.sqrt(3)/2, 1/2, b],
              [-np.sqrt(3)/2, 1/2, b]])/r

if __name__ == '__main__':
    
    data = {'Ac': Ac, 'Bc': Bc, 'Cc': Cc, 'Dc': Dc}
    savemat('data/cont_sys.mat', data)
