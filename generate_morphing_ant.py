import numpy as np
import argparse
from softlearning.environments.gym.mujoco.morphing_ant import DEFAULT_ANT
from softlearning.environments.gym.mujoco.morphing_ant import UPPER_BOUND
from softlearning.environments.gym.mujoco.morphing_ant import LOWER_BOUND
from softlearning.environments.gym.mujoco.morphing_ant import Leg
from examples.development.variants import ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK
from examples.instrument import run_example_local


if __name__ == '__main__':

    parser = argparse.ArgumentParser('MorphingAnt')
    parser.add_argument('--num_legs', type=int, default=4)
    parser.add_argument('--dataset_size', type=int, default=100)
    args = parser.parse_args()

    for i in range(args.dataset_size):

        legs = [Leg(*np.random.uniform(
            LOWER_BOUND, UPPER_BOUND)) for _ in range(args.num_legs)]
        if i == 0 and len(DEFAULT_ANT) == args.num_legs:
            legs = DEFAULT_ANT

        ENVIRONMENT_PARAMS_PER_UNIVERSE_DOMAIN_TASK[
            'gym']['MorphingAnt']['v0']['legs'] = legs

        run_example_local('examples.development', (
            '--algorithm', 'SAC',
            '--universe', 'gym',
            '--domain', 'MorphingAnt',
            '--task', 'v0',
            '--exp-name', f'morphing-ant-{i}',
            '--checkpoint-frequency', '1000',
            '--mode=local'))
