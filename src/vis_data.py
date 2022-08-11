import os
import subprocess

import cv2
import matplotlib.pyplot as plt
import numpy as np

from . import system_data as sysdat


def triple_plot(data, title, figsize=(8, 8)):
    """Plot the control signals"""
    fig = plt.figure(figsize=figsize)

    x = np.arange(len(data))
    ax = fig.add_subplot(311)
    ax.step(x, data[:, 0])
    plt.title("Motor 1")
    plt.grid(True)
    # ax.grid(color='k', linestyle='--', linewidth=1)
    ax = fig.add_subplot(312)
    ax.step(x, data[:, 1])
    plt.title("Motor 2")
    plt.grid(True)
    ax = fig.add_subplot(313)
    ax.step(x, data[:, 2])
    plt.title("Motor 3")
    #    plt.title(title)

    plt.grid(True)
    fig.suptitle(title)


def triple_plot2(data1, data2, title1, title2, figsize=(8, 8)):
    """Plot the states and the reference"""

    fig = plt.figure(figsize=figsize)
    x = np.arange(len(data1))

    ax = fig.add_subplot(311)
    ax.step(x, data1[:, 0], label=title1)
    ax.step(x, data2[:, 0], label=title2)
    plt.title(r"$v$")
    plt.legend()
    plt.grid(True)
    # ax.grid(color='k', linestyle='--', linewidth=1)
    ax = fig.add_subplot(312)
    ax.step(x, data1[:, 1], label=title1)
    ax.step(x, data2[:, 1], label=title2)
    plt.title(r"$v_n$")
    plt.legend()
    plt.grid(True)
    ax = fig.add_subplot(313)
    ax.step(x, data1[:, 2], label=title1)
    ax.step(x, data2[:, 2], label=title2)
    plt.title(r"$\omega$")
    # fig.suptitle(title)
    plt.legend()
    plt.grid(True)

    plt.suptitle("States")
    return fig


def plot_poles(poles, q, r, c="r", label="Poles"):

    fig = plt.figure()
    plt.scatter([x.real for x in poles], [x.imag for x in poles], c=c, label=label)
    plt.title("Poles")
    plt.legend()
    plt.grid()
    plt.xlabel("Real(pole)")
    plt.ylabel("Imag(pole)")
    ax = plt.gca()
    circle = plt.Circle((q, 0), radius=r, color="b", fill=False)
    ax.add_artist(circle)

    ax.set_xlim(
        (min(np.min(np.real(poles)), q - r), max(np.max(np.real(poles)), q + r))
    )
    ax.set_ylim((min(np.min(np.imag(poles)), -r), max(np.max(np.imag(poles)), r)))

    return fig


def scatter_controllers(arrs, titles, figsize=(8, 8)):

    N = np.arange(0, len(arrs[0]))
    fig, axs = plt.subplots(4, 1, figsize=figsize)
    for i, ax in enumerate(axs):
        ax.scatter(N, arrs[i])
        ax.set_title(titles[i])
        ax.grid()

    return fig


def plot_pose(pose, traj, size, figsize=(8, 8)):

    x = traj[:, 0]
    y = traj[:, 1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111)
    ax.plot(pose[:, 0], pose[:, 1], label="Trajectory")
    ax.scatter(x, y, marker="*", c="r", s=100, label="Checkpoints")
    ax.set_xlim((0, size))
    ax.set_ylim((0, size))
    fig.suptitle("Trajectory")
    plt.legend()
    plt.grid(True)
    return fig


class SimulationDrawer:
    def __init__(self, pose_vec, state_vec, image_wh=2048, field_wh=10):

        self.pose_vec = pose_vec
        self.state_vec = state_vec
        self.robot_radius = sysdat.b
        self.tire_radius = sysdat.r
        self.Ts = sysdat.Ts
        self.FPS = 1 / self.Ts
        # canvas and field are squares
        # self.canvas_wh = canvas_wh  # pixels
        self.field_wh = field_wh  # meters
        self.image_wh = image_wh  # pixels

    @property
    def conv_factor(self):
        return self.image_wh / self.field_wh

    # def _cy(self, y):
    #     """Convert y coordinate to canvas coordinate"""
    #     return FIELD_DIM - y

    def cp(self, point):
        """convert point from field to canvas coords"""
        return [
            int(point[0] * self.conv_factor),
            int(self.image_wh - self.conv_factor * point[1]),
        ]

    def rotate(self, vector, pose):

        theta = pose[2]
        rot = np.array(
            [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
        )
        return np.dot(rot, vector)

    def draw_robot(self, frame, pose):
        robot_center_canvas = self.cp(pose[:2])
        frame = cv2.circle(
            frame,
            robot_center_canvas,
            int(self.robot_radius * self.conv_factor),
            (0, 255, 0),
            -1,
        )

        vector_front_robot = np.array([self.robot_radius, 0])
        vector_front_global = self.rotate(vector_front_robot, pose)
        front_canvas = self.cp(
            [pose[0] + vector_front_global[0], pose[1] + vector_front_global[1]]
        )
        frame = cv2.circle(
            frame,
            front_canvas,
            int(self.robot_radius * self.conv_factor / 5),
            (255, 0, 0),
            -1,
        )

        return frame

    def draw_state(self, frame, pose, state):

        v_vector_orig = np.array([state[0], 0]) / 2
        v_vector_global = self.rotate(v_vector_orig, pose)

        frame = cv2.arrowedLine(
            frame,
            self.cp(pose[:2]),
            self.cp([pose[0] + v_vector_global[0], pose[1] + v_vector_global[1]]),
            (0, 0, 255),
            1,
        )

        vn_vector_orig = np.array([0, state[1]]) / 2
        vn_vector_global = self.rotate(vn_vector_orig, pose)
        frame = cv2.arrowedLine(
            frame,
            self.cp(pose[:2]),
            self.cp([pose[0] + vn_vector_global[0], pose[1] + vn_vector_global[1]]),
            (0, 0, 255),
            1,
        )

        return frame

    def draw_path(self, frame, pose_vec_slice):

        for idx in range(1, len(pose_vec_slice)):
            frame = cv2.arrowedLine(
                frame,
                self.cp(pose_vec_slice[idx - 1][:2]),
                self.cp(pose_vec_slice[idx][:2]),
                (0, 0, 255),
                1,
            )
        return frame

    def generate_start_frame(self):
        grid_dist = int(self.conv_factor * 1)
        frame_template = (255 * np.ones((self.image_wh, self.image_wh, 3))).astype(
            np.uint8
        )

        for start in range(0, self.image_wh, grid_dist):
            frame_template = cv2.line(
                frame_template, (start, 0), (start, self.image_wh), (0, 0, 0), 1
            )
            frame_template = cv2.line(
                frame_template, (0, start), (self.image_wh, start), (0, 0, 0), 1
            )

        return frame_template

    def add_status_pane(self, frame, pose, state):

        background = np.zeros((self.image_wh, self.image_wh // 5, 3)).astype(np.uint8)

        background = cv2.putText(
            background,
            "V: {:.2f} m/s".format(state[0]),
            (100, 200),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3,
        )

        background = cv2.putText(
            background,
            "Vn: {:.2f} m/s".format(state[1]),
            (100, 400),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3,
        )

        background = cv2.putText(
            background,
            "w: {:.2f} rad/s".format(state[2]),
            (100, 600),
            cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1,
            color=(255, 255, 255),
            thickness=3,
        )

        out = np.zeros((self.image_wh, self.image_wh + self.image_wh // 5, 3)).astype(
            np.uint8
        )
        out[:, : self.image_wh, :] = frame
        out[:, self.image_wh :, :] = background
        return out.astype(np.uint8)

    def generate_video(self):

        os.makedirs("temp", exist_ok=True)
        # frame_template = (255 * np.ones((self.image_wh, self.image_wh, 3))).astype(
        #     np.uint8
        # )
        frame_template = self.generate_start_frame()
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        temp_video_path = "temp/sim_video.mp4"
        conv_video_path = "temp/sim_video_conv.mp4"

        # writer = cv2.VideoWriter(
        #     temp_video_path,
        #     fourcc,
        #     self.FPS,
        #     (self.image_wh, self.image_wh),
        # )

        writer = cv2.VideoWriter(
            temp_video_path,
            fourcc,
            self.FPS,
            (self.image_wh + self.image_wh // 5, self.image_wh),
        )
        for idx, (pose, state) in enumerate(zip(self.pose_vec, self.state_vec)):
            # print("Here")
            frame = frame_template.copy()
            frame = self.draw_path(frame, self.pose_vec[:idx])
            frame = self.draw_robot(frame, pose)
            frame = self.draw_state(frame, pose, state)
            frame = self.add_status_pane(frame, pose, state)
            # print(frame.shape)
            writer.write(frame)

        # print(writer)
        writer.release()

        subprocess.call(
            args=f"ffmpeg -y -i {temp_video_path} -c:v libx264 {conv_video_path}".split(
                " "
            )
        )

        return conv_video_path
