import numpy as np
from numpy.random import Generator, default_rng


def create_track_distribution(
    computation_time_steps: int,
    dt: float,
    number_of_tracks: int,
    separation_time_steps: int,
    random_generator: Generator = default_rng(2137),
):
    # Time when the tracks can be generated
    beam_generation_time = (computation_time_steps - separation_time_steps) * dt

    # Generate a random delay for each beam (between 0, 1)
    track_delay = random_generator.random(number_of_tracks)

    # Make the delays relative to each other (add the sum of previous delays to each element)
    summed_delays = np.cumsum(track_delay)

    # Normalize the delays to fit in the beam generation time

    # two implementations: 1st is the original, it's biased towards the end of the beam generation
    # time since it normalizes via the last element, the 2nd implementation normalizes by the
    # number of beams - this ensures a uniform distribution of beams over the beam generation time
    # TODO: check if the 1st implementation bias was intentional or accidental

    # normalized_delays = summed_delays / summed_delays[-1] * beam_generation_time
    normalized_delays = summed_delays / number_of_tracks * beam_generation_time

    # create a bin for each time step
    bins = np.arange(0.0, beam_generation_time + dt, dt)

    # count the delays by the bin they fall into
    initialise_tracks = np.asarray(
        np.histogram(normalized_delays, bins)[0], dtype=np.int32
    )

    # append the zeros to the track distribution to match the computation time steps
    dont_initialise_tracks = np.zeros(separation_time_steps)
    track_times_histrogram = np.asarray(
        np.concatenate((initialise_tracks, dont_initialise_tracks)), dtype=np.int32
    )

    return track_times_histrogram
