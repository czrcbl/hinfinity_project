import numpy as np
import matplotlib.pyplot as plt


def triple_plot(data, title, figsize=(8, 8)):
    
    fig = plt.figure(figsize=figsize)
        
    x = np.arange(len(data))
    ax = fig.add_subplot(311)
    ax.step(x, data[:, 0])
    plt.grid(True)
    # ax.grid(color='k', linestyle='--', linewidth=1)
    ax = fig.add_subplot(312)
    ax.step(x, data[:, 1])
    plt.grid(True)
    ax = fig.add_subplot(313)
    ax.step(x, data[:, 2])
    plt.title(title)
    plt.grid(True)


def triple_plot2(data1, data2, title1, title2, figsize=(8, 8)):
    

    fig = plt.figure(figsize=figsize)
    
    x = np.arange(len(data1))
    ax = fig.add_subplot(311)
    ax.step(x, data1[:, 0], label=title1)
    ax.step(x, data2[:, 0], label=title2)
    plt.legend()
    plt.grid(True)
    # ax.grid(color='k', linestyle='--', linewidth=1)
    ax = fig.add_subplot(312)
    ax.step(x, data1[:, 1], label=title1)
    ax.step(x, data2[:, 1], label=title2)
    plt.legend()
    plt.grid(True)
    ax = fig.add_subplot(313)
    ax.step(x, data1[:, 2], label=title1)
    ax.step(x, data2[:, 2], label=title2)
    # fig.suptitle(title)
    plt.legend()
    plt.grid(True)


def plot_poles(poles, q, r, c='r', label='Poles'):
    
    plt.scatter([x.real for x in poles], [x.imag for x in poles], c=c, label=label)
    plt.title('Poles')
    plt.legend()
    plt.grid()
    plt.xlabel('Real(pole)')
    plt.ylabel('Imag(pole)')
    ax = plt.gca()
    circle = plt.Circle((q, 0), radius=r, color='b', fill=False)
    ax.add_artist(circle)
    
    ax.set_xlim((min(np.min(np.real(poles)), q-r), max(np.max(np.real(poles)), q+r)))
    ax.set_ylim((min(np.min(np.imag(poles)), -r), max(np.max(np.imag(poles)), r)))
    

def scatter_controllers(arrs, titles, figsize=(8, 8)):
    
    N = np.arange(0, len(arrs[0]))
    fig, axs = plt.subplots(4, 1, figsize=figsize)
    for i, ax in enumerate(axs):
        ax.scatter(N, arrs[i])
        ax.set_title(titles[i])
        ax.grid()


def plot_pose(pose, traj, figsize=(8, 8)):
    
    x = traj[:, 0]
    y = traj[:, 1]
    
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(pose[:, 0], pose[:, 1], label='Trajectory')
    ax.scatter(x, y, marker='*', c='r', s=100, label='Checkpoints')
    fig.suptitle('Trajectory')
    plt.legend()
    plt.grid(True)


if __name__ == '__main__':
    pass
