from morphing_agents.mujoco.dog.env import MorphingDogEnv
from morphing_agents.mujoco.dog.designs import DEFAULT_DESIGN


import numpy as np
import argparse
import skvideo.io


if __name__ == '__main__':

    parser = argparse.ArgumentParser('MorphingDog')
    parser.add_argument('--num-legs', type=int, default=4)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--episode-length', type=int, default=100)
    args = parser.parse_args()

    frames = []

    e = MorphingDogEnv(fixed_design=DEFAULT_DESIGN)
    e.reset()
    for i in range(args.episode_length):
        o, r, d, _ = e.step(e.action_space.sample())
        frames.append(e.render(mode='rgb_array'))
        if d:
            break

    e = MorphingDogEnv(num_legs=args.num_legs)
    for n in range(args.num_episodes):
        e.reset()
        for i in range(args.episode_length):
            o, r, d, _ = e.step(e.action_space.sample())
            frames.append(e.render(mode='rgb_array'))
            if d:
                break

    frames = np.array(frames)
    skvideo.io.vwrite("dog.mp4", frames)
