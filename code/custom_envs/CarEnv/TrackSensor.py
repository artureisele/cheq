import gymnasium as gym
import numpy as np
from shapely.geometry import MultiPoint, LinearRing, LineString, Point, Polygon

from .Sensor import Sensor
from .Utils import visualize_track_state


class TrackSensor(Sensor):
    def __init__(self, env, no_points=20, look_ahead_distance=60):
        super(TrackSensor, self).__init__(env)
        self.no_points = no_points
        self.look_ahead_distance = look_ahead_distance

    #@property
    #def bbox(self):
    #    return tuple(self._bbox)

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, shape=(self.no_points, 2))

    #@property
    #def view_normalizer(self):
    #    return max((abs(x) for x in self._bbox))

    def observe(self, env):

        centerline = env.problem.track_dict['centerline']
        x, y, theta = env.vehicle_model.get_pose(env.vehicle_state)[0][:3]

        position = np.array([x, y, 1])
        position = position[..., None]
        transformed = env.ego_transform @ position

        lr = LinearRing(centerline)
        car_point = Point(x, y)

        #negative_distance = lr.interpolate(-1)
        #positive_distance = lr.interpolate(-1 + lr.length)

        footpoint_distance = lr.project(car_point)
        point_distances = np.linspace(-self.look_ahead_distance, 0, self.no_points)[::-1]
        point_distances = point_distances + footpoint_distance
        points = [lr.interpolate(distance) for distance in point_distances]

        # visualize_track_state(env.problem.track_dict, x, y, theta, points=np.array([[point.x, point.y] for point in points]))

        points = np.array([[point.x, point.y, 1] for point in points])
        ego_points = np.squeeze(env.ego_transform @ points[..., None], -1)
        ego_points = ego_points[:, :2] / self.look_ahead_distance  # simple normalization

        # xy_distances = np.array([[car_point.x - point.x, car_point.y - point.y] for point in points])

        return ego_points
