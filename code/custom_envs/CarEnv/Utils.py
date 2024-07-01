import os
import pickle

import numpy as np
from matplotlib import pyplot as plt

from custom_envs import RACING_FAST
from custom_envs.CarEnv.Track.Generator import make_full_environment


def visualize_track_state(track, pos_x, pos_y, theta, points=None):
    centerline = track['centerline']
    #start_pos = track['start_xy']
    #start_orientation = track['start_theta']
    cones = track['cone_pos']

    # plot start position with orientation
    dx = 5 * np.cos(theta)
    dy = 5 * np.sin(theta)
    plt.plot(pos_x, pos_y, color='green', marker='o', markersize=10)
    plt.arrow(pos_x, pos_y, dx, dy, head_width=2, head_length=2, width=0.01, fc='black', ec='black')

    # plot additional points:
    if points is not None:
        plt.scatter(points[:, 0], points[:, 1], color='black', marker='x', s=100)

    # plot centerline
    track_x = centerline[:, 0]
    track_y = centerline[:, 1]
    plt.plot(track_x, track_y, color='red', alpha=0.95)

    # plot cones
    cones_x = cones[:, 0]
    cones_y = cones[:, 1]
    plt.scatter(cones_x, cones_y, color='black', s=2, alpha=0.75)

    plt.show()


def visualize_track(track):
    centerline = track['centerline']
    pos_x, pos_y = track['start_xy']
    theta = track['start_theta']
    cones = track['cone_pos']

    # plot start position with orientation
    dx = 5 * np.cos(theta)
    dy = 5 * np.sin(theta)
    plt.plot(pos_x, pos_y, color='green', marker='o', markersize=10)
    plt.arrow(pos_x, pos_y, dx, dy, head_width=2, head_length=2, width=0.01, fc='black', ec='black')

    # plot additional points:
    # if points is not None:
    #     plt.scatter(points[:, 0], points[:, 1], color='black', marker='x', s=100)

    # plot centerline
    track_x = centerline[:, 0]
    track_y = centerline[:, 1]
    plt.plot(track_x, track_y, color='red', alpha=0.95)

    # plot cones
    cones_x = cones[:, 0]
    cones_y = cones[:, 1]
    plt.scatter(cones_x, cones_y, color='black', s=2, alpha=0.75)

    plt.show()


def generate_tracks(num_tracks: int):
    tracks = []

    for i in range(num_tracks):
        track = make_full_environment(width=RACING_FAST['problem']['track_width'],
                                      extends=(RACING_FAST['problem']['extend'], RACING_FAST['problem']['extend']),
                                      cone_width=RACING_FAST['problem']['cone_width'],
                                      rng=np.random.default_rng())
        tracks += [track]

    return tracks


def load_track(path: str):
    with open(path, 'rb') as track_file:
        track = pickle.load(track_file)
    return track


def load_tracks(tracks_dir: str):
    tracks = []
    track_paths = os.listdir(tracks_dir)
    for track_path in track_paths:
        tracks.append(load_track(os.path.join(tracks_dir, track_path)))

    return tracks


def save_track(track, save_path: str):
    with open(save_path, 'wb') as track_file:
        pickle.dump(track, track_file)


def save_tracks(tracks, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    for idx, track in enumerate(tracks):
        save_track(track, os.path.join(save_dir, f"track{idx:02d}.obj"))


def visualize_track_from_file(path: str):
    track = load_track(path)
    visualize_track(track)


if __name__ == '__main__':
    #tracks = generate_tracks(10)
    #save_tracks(tracks, 'SavedTracks/10_test_tracks_extra')
    tracks = load_tracks('SavedTracks/10_test_tracks_extra')
    for idx, track in enumerate(tracks):
        print('Map', idx)
        visualize_track(track)