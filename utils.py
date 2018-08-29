import numpy as np
import pickle


def save(file_name, obj):
    
    with open(file_name, 'wb') as f:
        pickle.dump(obj, f)
        
        
def load(file_name):
    
    with open(file_name, 'rb') as f:
        obj = pickle.load(f)
        
    return obj


def make_ref(ref, N):
    return np.tile(ref, (N, 1))


def make_inv_ref(ref1, ref2,  N):
    a = np.tile(ref1, (int(N/2), 1))
    b = np.tile(ref2, (int(N/2), 1))

    return np.vstack((a, b))


def get_clpoles(Ad, Bd, K): 

    Z = np.zeros((3,3));
    I = np.eye(3);
    B2 = np.block([[Bd] , [Z]]);
    A = np.block([[Ad, Z], [I, I]])

    return np.linalg.eig(A + B2.dot(K))[0]


class StepInfoData:
    
    def __init__(self, **kargs):
        self.__dict__.update(kargs)
    
    def __str__(self):
        s = ''
        for key, val in self.__dict__.items():
            s = s + '%s: %.3f\n' % (key, val)
        
        return s[:-1]
    
    def __repr__(self):
        s = 'StepInfoData('
        for key, val in self.__dict__.items():
            s = s + key + '=' + str(val) + ', '

        return s[:-2] + ')'
            
def step_info(time, y):
    
    assert len(time) == len(y), 'signal and time must have the same length'
    
    for i in range(len(y)):
        s = y[i:]
        if len(s) == 1:
            sval = s[0]
            st = time[i]
        else:
            d = np.min(s) + (np.max(s) - np.min(s))/2
            if (d * 1.02 >= np.max(s)) and (d * 0.98 <= np.min(s)):
                sval = d
                st = time[i]
                break
    
    os = (np.max(y) - sval)/sval
    pt = time[np.argmax(y)]
    
    idx = np.where((y >= 0.1 * sval) & (y <= 0.9 * sval))[0] 
    if len(idx) == 0:
        rt = float('NaN')
    else:
        io = idx[0]
        for i in idx[1:]:
            if i != io + 1:
                break
            io = i
        
        rt = time[io] - time[idx[0]]
    
    data = StepInfoData(RiseTime=rt, SettlingTime=st, Overshoot=os, 
                        PeakTime=pt)
    
    return data


def load_square_traj():
    """One meter side, 20cm spaced marks"""
    traj = np.array([
                    [0, 0, 0],
                    [0.2, 0, 0],
                    [0.4, 0, 0],
                    [0.6, 0, 0],
                    [0.8, 0, 0],
                    [1, 0, 0],
                    [1, 0.2, 0],
                    [1, 0.4, 0],
                    [1, 0.6, 0],
                    [1, 0.8, 0],
                    [1, 1, 0],
                    [0.8, 1, 0],
                    [0.6, 1, 0],
                    [0.4, 1, 0],
                    [0.2, 1, 0],
                    [0, 1, 0],
                    [0, 0.8, 0],
                    [0, 0.6, 0],
                    [0, 0.4, 0],
                    [0, 0.2, 0],
                    [0, 0, 0]
                    ])
    return traj


def load_8_traj():

    angles = np.array([0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
                       180])
    angles1 = np.hstack((angles, np.flip(-angles[:-1], 0)))
    angles1 = np.pi/180*angles1
    angles2 = np.hstack((np.flip(angles, 0), -angles[1:]))
    angles2 = np.pi/180 * angles2
    circle1 = np.zeros(shape=(len(angles1), 3))
    circle1[:, 0] = np.cos(angles1)
    circle1[:, 1] = np.sin(angles1)

    circle2 = np.zeros(shape=(len(angles2), 3))
    circle2[:, 0] = np.cos(angles2)
    circle2[:, 1] = np.sin(angles2)

    circle1 = circle1 + np.array([1, 0, 0])
    circle2 = circle2 + np.array([3, 0, 0])
    traj = np.vstack((np.array([0, 0, 0]), circle2[0:-1], circle1))

    return traj


def load_circle_J_traj():
    from scipy.io import loadmat
    
    return loadmat('data/circle_traj_J.mat')['traj']


if __name__ == '__main__':
    pass