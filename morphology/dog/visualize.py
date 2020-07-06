from softlearning.environments.gym.mujoco.morphing_dog import DEFAULT_DOG
from softlearning.environments.gym.mujoco.morphing_dog import UPPER_BOUND
from softlearning.environments.gym.mujoco.morphing_dog import LOWER_BOUND
from softlearning.environments.gym.mujoco.morphing_dog import Leg
from softlearning.environments.gym.mujoco.morphing_dog import MorphingDogEnv


import numpy as np
import argparse
import skvideo.io


if __name__ == '__main__':

    parser = argparse.ArgumentParser('MorphingDog')
    parser.add_argument('--num_legs', type=int, default=4)
    parser.add_argument('--dataset_size', type=int, default=10)
    parser.add_argument('--episode_length', type=int, default=100)
    args = parser.parse_args()

    LEGS_SPEC = []
    for i in range(args.dataset_size):
        if i == 0 and len(DEFAULT_DOG) == args.num_legs:
            LEGS_SPEC.append(DEFAULT_DOG)
        else:
            LEGS_SPEC.append([Leg(*np.random.uniform(
                LOWER_BOUND, UPPER_BOUND)) for _ in range(args.num_legs)])

    frames = []
    for spec in LEGS_SPEC:
        e = MorphingDogEnv(legs=spec)
        e.reset()
        for i in range(args.episode_length):
            e.step(e.action_space.sample())
            frames.append(e.render(mode='rgb_array'))

    frames = np.array(frames)
    skvideo.io.vwrite("dog.mp4", frames)
