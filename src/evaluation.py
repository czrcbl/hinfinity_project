import numpy as np


class StepInfoData:
    def __init__(self, **kargs):
        self.__dict__.update(kargs)

    def __str__(self):
        s = ""
        for key, val in self.__dict__.items():
            s = s + "%s: %.3f\n" % (key, val)

        return s[:-1]

    def __repr__(self):
        s = "StepInfoData("
        for key, val in self.__dict__.items():
            s = s + key + "=" + str(val) + ", "

        return s[:-2] + ")"

    def todict(self):
        return {
            "RiseTime": float(self.RiseTime),
            "SettlingTime": float(self.SettlingTime),
            "Overshoot": float(self.Overshoot),
            "PeakTime": float(self.PeakTime),
        }

    def fromdict(self, d):
        self.RiseTime = d["RiseTime"]
        self.SettlingTime = d["SettlingTime"]
        self.Overshoot = d["Overshoot"]
        self.PeakTime = d["PeakTime"]


def step_info(time, y):

    assert len(time) == len(y), "signal and time must have the same length"
    assert len(y.shape) <= 2, "signal must be 1 or 2 dimensional"
    assert (len(y.shape) == 1) or (
        len(y) == y.shape[0] * y.shape[2]
    ), "2d array must be a row or column vector"

    # in case y be a np.matrix
    y = np.array(y).flatten()

    for i in range(len(y)):
        s = y[i:]
        if len(s) == 1:
            sval = s[0]
            st = time[i]
        else:
            d = np.min(s) + (np.max(s) - np.min(s)) / 2
            if (d * 1.02 >= np.max(s)) and (d * 0.98 <= np.min(s)):
                sval = d
                st = time[i]
                break

    os = (np.max(y) - sval) / sval
    pt = time[np.argmax(y)]

    idx = np.where((y >= 0.1 * sval) & (y <= 0.9 * sval))[0]
    if len(idx) == 0:
        rt = float("NaN")
    else:
        io = idx[0]
        for i in idx[1:]:
            if i != io + 1:
                break
            io = i

        rt = time[io] - time[idx[0]]

    data = StepInfoData(RiseTime=rt, SettlingTime=st, Overshoot=os, PeakTime=pt)

    return data
