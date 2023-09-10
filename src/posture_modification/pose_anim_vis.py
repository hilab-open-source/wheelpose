import os
from os.path import join as pjoin
from tqdm import tqdm
import numpy as np
import argparse

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.patheffects as PathEffects

KINEMATIC_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 22, 12, 15], [22, 14, 17, 19, 21], [22, 13, 16, 18, 20]]
KINEMATIC_CHAIN = [[0, 2, 5, 8, 11], [0, 1, 4, 7, 10], [0, 3, 6, 9, 22, 12, 15], [22, 14, 17], [17, 19, 21], [22, 13, 16], [16, 18, 20]]
SUBPLOT_VIEW_DICT = {221:(45,-45), 222:(0,-90), 223:(90,-90), 224:(0,0)}
BONE_COLORS = ['red', 'blue', 'black', 'red', 'blue',
            'darkblue', 'darkblue', 'darkblue', 'darkblue', 'darkblue',
            'darkred', 'darkred','darkred','darkred','darkred']
"""Quick script to graph Text2Motion data generated out for testing purposes
"""
def plot_3d_motion(save_path, joints, title, figsize=(24, 24), fps=120, radius=4, kinematic_tree=KINEMATIC_CHAIN, colors=BONE_COLORS):
    def plot_bones(ax, d):
        lines = []
        for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
#             print(color)
            if i < 5:
                linewidth = 2.0
            else:
                linewidth = 1.5

            line = ax.plot(d[chain, 0], d[chain, 1], d[chain, 2], linewidth=linewidth, color=color)
            lines.append(line)
        return lines

    # (seq_len, joints_num, 3)
    data = joints.copy().reshape(len(joints), -1, 3)

    joint_pos_mins = data.min(axis=0).min(axis=0)
    joint_pos_maxs = data.max(axis=0).max(axis=0)
    frame_number = data.shape[0]

    # moving the data down to center in graph
    height_offset = joint_pos_mins[1]
    data[:, :, 1] -= height_offset
    trajec = data[:, 0, [0, 2]]

    data[..., 0] -= data[:, 0:1, 0]
    data[..., 1] -= data[:, 0:1, 1] - radius/2
    data[..., 2] -= data[:, 0:1, 2] - radius/2

    title_sp = title.split(' ')
    if len(title_sp) > 10:
        title = '\n'.join([' '.join(title_sp[:10]), ' '.join(title_sp[10:])])


    # initializing the graph
    fig = plt.figure(figsize=figsize)
    fig.suptitle(title, fontsize=20)
    axs_lines = {}
    axs_scatter = {}
    axs_text = {}
    for subplot, ax_view in SUBPLOT_VIEW_DICT.items():
        ax = fig.add_subplot(subplot, projection="3d")
        ax.view_init(*ax_view)

        ax.set_xlim([-radius / 2, radius / 2])
        ax.set_ylim([0, radius])
        ax.set_zlim([0, radius])

        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_zticklabels([])

        ax.grid(visible=False)
        # ax.axis("off")
        axs_lines[ax] = plot_bones(ax, data[0])

        axs_scatter[ax] = ax.scatter(data[0,:,0], data[0,:,1], data[0,:,2], color="green", alpha=.7, s=50)

        axs_text[ax] = []
        for i in range(data.shape[1]):
            txt = ax.text(data[0,i,0], data[0,i,1], data[0,i,2], str(i),fontsize=7, color="b")
            txt.set_path_effects([PathEffects.withStroke(linewidth=3, foreground='w')])
            axs_text[ax].append(txt)

        origin = np.array([[0,0,0]for _ in range(3)])
        ax.quiver(*origin, [1,0,0], [0,1,0], [0,0,1], color=["r", "g", "b"])

    def update(index):
        for ax, line in axs_lines.items():
            for i, (chain, color) in enumerate(zip(kinematic_tree, colors)):
                line[i][0].set_data_3d(data[index, chain, 0], data[index, chain, 1], data[index, chain, 2])
        for ax, scatter in axs_scatter.items():
            scatter._offsets3d = (data[index,:,0], data[index,:,1], data[index,:,2])

        for ax, text in axs_text.items():
            for i, t in enumerate(text):
                t.set_x(data[index,i,0])
                t.set_y(data[index,i,1])
                t.set_z(data[index,i,2])

    ani = FuncAnimation(fig, update, frames=frame_number, interval=1000/fps, repeat=False)
    ani.save(save_path, fps=fps)
    plt.close()

def get_parser():
    parser = argparse.ArgumentParser(
        prog="Posture Visualization",
        description="Visualizes converted posture files."
    )

    parser.add_argument(
                        "-i",
                        "--input",
                        type=str,
                        help="input file of a .npy posture rotational file",
                        required=True
                        )
    parser.add_argument(
                        "-o",
                        "--output",
                        type=str,
                        help="output file. Must be a gif",
                        required=True
                        )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    data = np.load(args.input)

    plot_3d_motion(args.output, data, title=args.input, fps=20, radius=1.5)
