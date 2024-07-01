import gymnasium as gym
import numpy as np
from matplotlib import pyplot as plt
from shapely.geometry import MultiPoint, LinearRing, LineString, Point, Polygon

from .Sensor import Sensor
from .Utils import visualize_track_state

def normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ControllerSensor(Sensor):
    def __init__(self, env, target_speed=6):
        super(ControllerSensor, self).__init__(env)
        self.target_speed = target_speed
        #self.no_points = no_points
        #self.look_ahead_distance = look_ahead_distance

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, shape=(4,))

    def observe(self, env):

        centerline = env.problem.track_dict['centerline']
        x, y, theta = env.vehicle_model.get_pose(env.vehicle_state)[0][:3]
        wheelbase = env.vehicle_model.wheelbase
        v_x, v_y = env.vehicle_model.v_loc_[0] if env.vehicle_model.v_loc_ is not None else (0, 0)

        velocity = np.hypot(v_x, v_y)

        #position = np.array([x, y, 1])
        #position = position[..., None]
        #transformed = env.ego_transform @ position

        lr = LinearRing(centerline)
        car_point = Point(x, y)
        # get the point at the front axle going relative to the cog
        front_axle = Point(x + np.cos(theta) * wheelbase * 0.5, y + np.sin(theta) * wheelbase * 0.5)

        front_axle_distance = lr.project(front_axle)
        foot_point = lr.interpolate(front_axle_distance)
        distance_to_center = lr.distance(front_axle)

        vector_to_right = np.array([np.sin(theta), -np.cos(theta)])
        vector_to_fp = np.array([foot_point.x - front_axle.x, foot_point.y - front_axle.y])
        cross_track_error = np.sign(np.dot(vector_to_right, vector_to_fp)) * distance_to_center

        next_point = front_axle_distance - 0.2
        next_point = lr.interpolate(next_point)

        track_tangent_vector = [next_point.x - foot_point.x, next_point.y - foot_point.y]
        tangent_angle = np.arctan2(track_tangent_vector[1], track_tangent_vector[0])

        error_angle = normalize_angle(tangent_angle - theta)

        # compute the curvature at future point by fitting a polynomial curve around

        future_dist = 2
        future_point = front_axle_distance - future_dist
        future_point = lr.interpolate(future_point)

        next_point = front_axle_distance - future_dist - 2.0
        next_point = lr.interpolate(next_point)

        previous_point = front_axle_distance - future_dist + 2.0
        previous_point = lr.interpolate(previous_point)

        x_prime = 0.5 * (next_point.x - previous_point.x)
        x_pprime = next_point.x - 2 * future_point.x + previous_point.x
        y_prime = 0.5 * (next_point.y - previous_point.y)
        y_pprime = next_point.y - 2 * future_point.y + previous_point.y

        kappa = np.abs(x_prime * y_pprime - y_prime*x_pprime) / (np.sqrt((x_prime*x_prime + y_prime*y_prime)**3) + 1e-4)
        curvature_radius = 1 / (kappa + 1e-4)

        #visualize_track_state(env.problem.track_dict, x, y, theta, np.array([[previous_point.x, previous_point.y],
        #                                                                     [next_point.x, next_point.y]]))


        # xy_distances = np.array([[car_point.x - point.x, car_point.y - point.y] for point in points])

        return np.array([cross_track_error, error_angle, velocity, curvature_radius])

