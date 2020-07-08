from morphing_agents.mujoco.ant.env import MorphingAntEnv
from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN


import numpy as np
import argparse
import skvideo.io


if __name__ == '__main__':

    parser = argparse.ArgumentParser('MorphingAnt')
    parser.add_argument('--num-legs', type=int, default=4)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--episode-length', type=int, default=100)
    args = parser.parse_args()

    frames = []

    e = MorphingAntEnv(fixed_design=DEFAULT_DESIGN)
    e.reset()
    for i in range(args.episode_length):
        o, r, d, _ = e.step(e.action_space.sample())
        frames.append(e.render(mode='rgb_array'))
        if d:
            break

    e = MorphingAntEnv(num_legs=args.num_legs)
    for n in range(args.num_episodes):
        e.reset()
        for i in range(args.episode_length):
            o, r, d, _ = e.step(e.action_space.sample())
            frames.append(e.render(mode='rgb_array'))
            if d:
                break

    frames = np.array(frames)
    skvideo.io.vwrite("ant.mp4", frames)
