import copy

import numpy as np
from shapely.geometry import Polygon
from shapely.affinity import translate, rotate


def create_polygon_from_dimensions(l: float, w: float, h: float = 0.0) -> Polygon:
    half_length = l / 2
    half_width = w / 2
    # half_height = h / 2

    # Calculate the coordinates of the four corners of the bounding box
    corner_1 = (+ half_length, + half_width)
    corner_2 = (- half_length, + half_width)
    corner_3 = (- half_length, - half_width)
    corner_4 = (+ half_length, - half_width)

    return Polygon([corner_1, corner_2, corner_3, corner_4])


def create_trajectory(traj_info: np.ndarray) -> np.ndarray:
    traj_info = traj_info[:, [0, 1, 2, 6, 7, 8]]
    
    return traj_info

class Agent(object):
    def __init__(self, traj_info: np.ndarray, id: int = None):
        self._id = id
        # self._data = traj_info # [91, 11] 
        # [x, y, z, l, w, h, heading, v_x, v_y, valid, type]
        self._x = traj_info[10, 0]
        self._y = traj_info[10, 1]
        self._z = traj_info[10, 2]
        self._l = traj_info[10, 3]
        self._w = traj_info[10, 4]
        self._h = traj_info[10, 5]
        self._type = traj_info[10, 10]
        
        self._polygon = create_polygon_from_dimensions(self._l, self._w)
        self._trajectory = traj_info[:, [0, 1, 2, 6, 7, 8]] # [91, 6] # [x, y, z, yaw, v_x, v_y]
        self._validity = traj_info[:, 9]
    
    @property
    def id(self) -> int:
        return self._id
    
    @property
    def type(self) -> int:
        return self._type
    
    @property
    def polygon(self) -> Polygon:
        return self._polygon
    
    @property
    def trajectory(self) -> np.ndarray:
        return self._trajectory
    
    @property
    def validity(self) -> np.ndarray:
        return self._validity
    
    def is_valid_at_t(self, t: int = 10) -> bool:
        return self._validity[t] == 1
    
    def get_position_at_t(self, t: int = 10) -> tuple:
        if self.is_valid_at_t(t):
            return self._trajectory[t, 0], self._trajectory[t, 1]
        else:
            return None
    
    def get_yaw_at_t(self, t: int = 10) -> float:
        if self.is_valid_at_t(t):
            return self._trajectory[t, 3]
        else:
            return None
    
    def get_speed_at_t(self, t: int = 10) -> float:
        if self.is_valid_at_t(t):
            return np.hypot(self._trajectory[t, -2], self._trajectory[t, -1])
        else:
            return None
    
    def get_polygon_at_t(self, t: int = 10) -> Polygon:
        if self.is_valid_at_t(t):
            x, y = self.get_position_at_t(t)
            yaw = self.get_yaw_at_t(t)
            polygon = copy.deepcopy(self._polygon)
            polygon = rotate(polygon, yaw, origin='center', use_radians=True)
            polygon = translate(polygon, xoff=x, yoff=y, zoff=0.0)
            return polygon
        else:
            return None
        
    