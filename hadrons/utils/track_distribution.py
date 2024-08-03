from random import Random

import numpy as np


def create_track_distribution(
    computation_time_steps: int,
    dt: float,
    number_of_tracks: int,
    separation_time_steps: int,
    random_generator: Random = Random(2137),
):
    time_s = computation_time_steps * dt
    randomized = random_generator.random(number_of_tracks)
    summed = np.cumsum(randomized)
    distributed_times = np.asarray(summed) / max(summed) * time_s
    bins = np.arange(0.0, time_s + dt, dt)
    initialise_tracks = np.asarray(
        np.histogram(distributed_times, bins)[0], dtype=np.int32
    )
    dont_initialise_tracks = np.zeros(separation_time_steps)
    track_times_histrogram = np.asarray(
        np.concatenate((initialise_tracks, dont_initialise_tracks)), dtype=np.int32
    )

    return track_times_histrogram
