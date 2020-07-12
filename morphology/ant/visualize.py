from morphing_agents.mujoco.ant.env import MorphingAntEnv
from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN
from morphing_agents.mujoco.ant.elements import LEG_UPPER_BOUND
from morphing_agents.mujoco.ant.elements import LEG_LOWER_BOUND
from morphing_agents.mujoco.ant.elements import LEG


import numpy as np
import argparse
import skvideo.io


if __name__ == '__main__':

    parser = argparse.ArgumentParser('MorphingAnt')
    #parser.add_argument('--num-legs', type=int, default=4)
    parser.add_argument('--num-episodes', type=int, default=10)
    parser.add_argument('--episode-length', type=int, default=100)
    args = parser.parse_args()

    frames = []

    ub = np.array(list(LEG_UPPER_BOUND))
    lb = np.array(list(LEG_LOWER_BOUND))
    scale = (ub - lb) / 2

    e = MorphingAntEnv(fixed_design=DEFAULT_DESIGN)
    e.reset()
    for i in range(args.episode_length):
        o, r, d, _ = e.step(e.action_space.sample())
        #frames.append(e.render(mode='rgb_array'))
        e.render(mode='human')

    for n in range(args.num_episodes):
        e = MorphingAntEnv(fixed_design=[
            LEG(*np.clip(np.array(
                leg) + np.random.normal(0, scale / 8), lb, ub))
            for leg in DEFAULT_DESIGN])
        e.reset()
        for i in range(args.episode_length):
            o, r, d, _ = e.step(e.action_space.sample())
            #frames.append(e.render(mode='rgb_array'))
            e.render(mode='human')

    frames = np.array(frames)
    skvideo.io.vwrite("ant.mp4", frames)
