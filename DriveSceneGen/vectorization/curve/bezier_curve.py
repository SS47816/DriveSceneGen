import numpy as np
import bezier


def fit_bezier_curve(n1, n2, spacing: float=1.0) -> np.ndarray:
    dist = np.hypot(n2[0] - n1[0], n2[1] - n1[1]) / 3
    n1_yaw = n1[2]
    n2_yaw = n2[2]
    p0 = (n1[0], n1[1])
    p1 = (n1[0] + np.cos(n1_yaw)*dist, n1[1] + np.sin(n1_yaw)*dist)
    p2 = (n2[0] - np.cos(n2_yaw)*dist, n2[1] - np.sin(n2_yaw)*dist)
    p3 = (n2[0], n2[1])

    nodes = np.array([p0, p1, p2, p3])
    curve = bezier.Curve(nodes.T, degree=3)

    s = np.linspace(0.0, 1.0, max(2, int(curve.length/spacing)))
    points = curve.evaluate_multi(s)
    points = points.T
    dx = np.diff(points[:, 0])
    dy = np.diff(points[:, 1])
    yaw = np.arctan2(dy, dx)

    end_tangent = curve.evaluate_hodograph(1.0)
    end_tangent = end_tangent.T.flatten()
    end_yaw = np.arctan2(end_tangent[1], end_tangent[0])
    ds = s[-1] - s[-2]
    dx = np.append(dx, ds*np.cos(end_yaw))
    dy = np.append(dy, ds*np.sin(end_yaw))
    yaw = np.append(yaw, end_yaw)
    s = s*curve.length

    return np.stack((points[:, 0], points[:, 1], yaw, dx, dy, s), axis=-1)