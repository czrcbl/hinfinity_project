from dataclasses import field
from operator import attrgetter
from re import L
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import pandas as pd
from PIL import Image
from src.simulation import trajectory_sim, Simulation
import src.system_data as sysdat
from src.utils import load
from src.project import Controller

# Specify canvas parameters in application
import numpy as np
import json
from pathlib import Path
from src.vis_data import plot_pose, triple_plot2, plot_poles
from src import system_data as sysdat
import cv2
import os
import subprocess

FIELD_WH = 5
CANVAS_WH = 500

TITTLE = "# 🎮 Play with the Simulation"


@st.cache
def load_controllers():

    import sys

    sys.path.insert(0, str(Path(__file__).absolute().parents[1] / "src"))
    all_controllers = load("data/controllers_solver-SCS_all.pkl")
    return all_controllers


def generate_start_frame(field_wh):

    WH = 1000
    grid_dist = int(WH / field_wh)
    frame_template = (255 * np.ones((WH, WH, 3))).astype(np.uint8)

    for start in range(0, WH, grid_dist):
        frame_template = cv2.line(frame_template, (start, 0), (start, WH), (0, 0, 0), 1)
        frame_template = cv2.line(frame_template, (0, start), (WH, start), (0, 0, 0), 1)

    return frame_template


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


def objects2points(objects, CONV_FACTOR):

    points = []
    for idx, line in objects.iterrows():
        if line.type == "line":
            points.append(
                [
                    (line.left + line.x1) * CONV_FACTOR,
                    (line.top + line.y1) * CONV_FACTOR,
                    0,
                ]
            )
            points.append(
                [
                    (line.left + line.x2) * CONV_FACTOR,
                    (line.top + line.y2) * CONV_FACTOR,
                    0,
                ]
            )
            points.insert(
                1,
                [
                    (points[0][0] + points[1][0]) / 2,
                    (points[0][1] + points[1][1]) / 2,
                    0,
                ],
            )
        elif line.type == "circle" and line.radius < 10:
            points.append([line.left * CONV_FACTOR, line.top * CONV_FACTOR, 0])
        elif line.type == "rect":
            points.append([line.left * CONV_FACTOR, line.top * CONV_FACTOR, 0])
            points.append(
                [(line.left + line.width) * CONV_FACTOR, line.top * CONV_FACTOR, 0]
            )
            points.append(
                [
                    (line.left + line.width) * CONV_FACTOR,
                    (line.top + line.height) * CONV_FACTOR,
                    0,
                ]
            )
            points.append(
                [line.left * CONV_FACTOR, (line.top + line.height) * CONV_FACTOR, 0]
            )
        elif line.type == "circle":
            for angle in range(-180, 180, 30):
                points.append(
                    [
                        (
                            line.left
                            + line.radius
                            + line.radius * np.cos(angle * np.pi / 180)
                        )
                        * CONV_FACTOR,
                        (line.top + line.radius * np.sin(angle * np.pi / 180))
                        * CONV_FACTOR,
                        0,
                    ]
                )
        elif line.type == "path":
            # st.write(line.path)
            for point in json.loads(line.path.replace("'", '"'))[::3]:
                # st.write(point)
                points.append([point[1] * CONV_FACTOR, point[2] * CONV_FACTOR, 0])

    return points


@st.cache(allow_output_mutation=True)
def do_simulation(traj, v_nav, sim, K):

    states_vec, control_signal_vec, ref_vec, pose_vec = trajectory_sim(
        traj, v_nav, sim, K
    )
    return states_vec, control_signal_vec, ref_vec, pose_vec


def main():
    global FIELD_WH, CANVAS_WH

    st.markdown(TITTLE)
    st.sidebar.markdown(TITTLE)

    FIELD_WH = st.sidebar.number_input(
        "Field Width and Height", value=FIELD_WH, min_value=3, max_value=10
    )
    drawing_mode = st.sidebar.selectbox(
        "Drawing tool:", ("freedraw", "point", "line", "rect", "circle")
    )

    stroke_width = 1
    point_display_radius = 3
    stroke_color = "#000000"
    bg_color = "#FFFFFF"
    realtime_update = True

    # Create a canvas component

    st.markdown(
        """
        ### Path Drawing Canvas

        Draw the robot path, the side of the canvas has a dimension of 10 meters.

        Choose the drawing tool on the side bar, the drawing can be reset on the trash can below the canvas.
    """
    )
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        background_image=Image.fromarray(generate_start_frame(FIELD_WH)),
        update_streamlit=realtime_update,
        height=CANVAS_WH,
        width=CANVAS_WH,
        drawing_mode=drawing_mode,
        point_display_radius=point_display_radius if drawing_mode == "point" else 0,
        display_toolbar=True,
        key="canvas",
    )
    tab_controller, tab_video, tab_plots = st.tabs(
        ["Controller Data", "Simulation Video", "Simulation Plots"]
    )
    # st.session_state["canvas_result"] = canvas_result.json_data

    # Do something interesting with the image data and paths
    # if canvas_result.image_data is not None:
    #     st.image(canvas_result.image_data)
    if canvas_result.json_data is not None:
        objects = pd.json_normalize(
            canvas_result.json_data["objects"]
        )  # need to convert obj to str because PyArrow
        for col in objects.select_dtypes(include=["object"]).columns:
            objects[col] = objects[col].astype("str")
        # st.dataframe(objects)

    st.sidebar.markdown(
        """
        ### Simulation parameters
    """
    )

    v_nav = st.sidebar.number_input("Reference Velocity (m/s)", value=0.6)

    all_controllers = load_controllers()
    controllers = [c for c in all_controllers if not c.poles is None]
    settling_time = np.array([c.stepinfo.SettlingTime for c in controllers])
    over_shoot = np.array([c.stepinfo.Overshoot for c in controllers])
    u_var = np.array([c.u_max_var for c in controllers])
    norm = np.array([c.norm for c in controllers])

    st.sidebar.markdown(
        """
        #### Controller Selection Parameters

        Here follows a list of many projected controllers for various center and radius configurations, you can select one based on its performance to use for the simulation.
    """
    )
    # cols = st.sidebar.columns(3)
    max_st = st.sidebar.number_input("Max settling time (s)", value=settling_time.max())
    max_norm = st.sidebar.number_input("Max norm", value=norm.max())
    max_u_var = st.sidebar.number_input(
        "Max Control Signal Variation (V)", value=u_var.max()
    )

    idx = np.where((settling_time < max_st) & (u_var < max_u_var) & (norm < max_norm))[
        0
    ]
    filt_controlers = sorted([controllers[ix] for ix in idx], key=lambda c: c.norm)

    c = st.sidebar.selectbox(
        "Filtered Controllers",
        filt_controlers,
        format_func=lambda c: f"Norm: {c.norm:.2f}, Center: {c.q:.2f}, Radius: {c.r:.2f}",
    )

    tab_controller.text(str(c))

    # tab_controller.write(
    #     f"Norm: {c.norm:.2f} - SettlingTime: {c.stepinfo.SettlingTime:.2f} - Overshoot: {c.stepinfo.Overshoot:.2f} - Umax_var: {c.u_max_var:.2f}"
    # )
    # c = controllers[idx[-1]]

    tab_controller.markdown(
        "The blue circle represents the constraint on the poles position."
    )
    fig = plot_poles(c.poles, c.q, c.r)
    tab_controller.pyplot(fig)

    if not st.sidebar.button("Start Simulation"):
        tab_video.warning("Simulation not started, press Start Simulation button")
        tab_plots.warning("Simulation not started, press Start Simulation button")
        return

    if len(objects) == 0:
        st.sidebar.warning("Draw Trajectory First")
        return

    # st.markdown(
    #     """
    #     ## Results
    # """
    # )

    with st.spinner():
        points = objects2points(objects, FIELD_WH / CANVAS_WH)

        sim = Simulation(sysdat.csys, sysdat.Ts)

        traj = np.array(points)
        traj[:, 1] = FIELD_WH - traj[:, 1]

        # st.write(traj)

        states_vec, control_signal_vec, ref_vec, pose_vec = do_simulation(
            traj, v_nav, sim, c.K
        )

        sim_drawer = SimulationDrawer(
            pose_vec, states_vec, image_wh=2048, field_wh=FIELD_WH
        )

        conv_video_path = sim_drawer.generate_video()

    with tab_video:
        st.markdown(
            """
            ## Robot Movement Video """
        )

        st.video(conv_video_path, format="MPEG-4")

    with tab_plots:

        cols = st.columns(2)
        fig = plot_pose(pose_vec, traj, size=FIELD_WH)
        cols[0].pyplot(fig)

        fig = triple_plot2(states_vec, ref_vec, "States", "Reference", figsize=(8, 10))
        cols[1].pyplot(fig)


main()
