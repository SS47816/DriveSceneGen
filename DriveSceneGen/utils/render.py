import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import axes
from matplotlib.patches import Rectangle
from PIL import Image


def render_vectorized_scenario_on_axes(
    ax: axes, lanes: list, agents: list, map_range: float = 80.0
):
    for lane in lanes:
        lane_np = np.array(lane)
        # Thick lane boundaries
        ax.plot(
            lane_np[:, 0],
            lane_np[:, 1],
            "slategrey",
            linestyle="solid",
            linewidth=40,
            alpha=1.0,
            solid_capstyle="round",
            zorder=1,
        )  # 'paleturquoise', 'aquamarine', 'palegreen', 'lightcyan'
        # Thin centerline
        ax.plot(
            lane_np[:, 0],
            lane_np[:, 1],
            "springgreen",
            linestyle="solid",
            linewidth=1,
            alpha=1.0,
            solid_capstyle="round",
            zorder=5,
        )  # 'paleturquoise', 'aquamarine', 'palegreen', 'lightcyan'
        # Thin arrows
        if lane_np.shape[1] <= 2:
            continue
        ax.quiver(
            lane_np[::20, 0],
            lane_np[::20, 1],
            lane_np[::20, 3] * 1.2,
            lane_np[::20, 4] * 1.2,
            color="springgreen",  # "#45FFCA", random_color(),
            angles="xy",
            scale_units="xy",
            units="xy",
            scale=1.0,
            zorder=50,
        )
    for agent in agents:
        # print(f"agent: {agent}")
        rect = Rectangle(
            (agent[0] - agent[3] / 2, agent[1] - agent[4] / 2),
            agent[3],
            agent[4],
            transform=mpl.transforms.Affine2D().rotate_around(
                agent[0], agent[1], agent[6]
            )
            + ax.transData,
            facecolor="#FF6969",  # '#FF6969', "#D67BFF",
            alpha=1.0,
            linewidth=2,
            zorder=100,
        )
        ax.add_patch(rect)

    agent_np = np.array(agents).reshape((-1, 9))
    ax.quiver(
        agent_np[:, 0],
        agent_np[:, 1],
        agent_np[:, -2] * 2.0,
        agent_np[:, -1] * 2.0,
        color="red",
        angles="xy",
        scale_units="xy",
        units="xy",
        scale=1.0,
        zorder=150,
    )

    margin = map_range / 2
    ax.axis([-margin, margin, -margin, margin])
    ax.set_aspect("equal")
    ax.margins(0)
    ax.grid(False)
    ax.axis("off")

    return ax
