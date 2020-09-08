from morphing_agents.mujoco.dkitty.designs import DEFAULT_DESIGN
from morphing_agents.mujoco.dkitty.env import MorphingDKittyEnv
from examples.instrument import generate_experiment_kwargs
from softlearning.environments.utils import get_environment_from_params
from softlearning import policies
import argparse
import importlib
import pickle as pkl
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser('DKittyOracle')
    parser.add_argument('--ckpt', type=str, default='./dkitty-oracle')
    args = parser.parse_args()

    module = importlib.import_module('examples.development')
    example_args = module.get_parser().parse_args((
        '--algorithm', 'SAC',
        '--universe', 'gym',
        '--domain', 'MorphingDKitty',
        '--task', 'v0',
        '--exp-name', f'dkitty-oracle',
        '--checkpoint-frequency', '10',
        '--mode=local',
        '--local-dir', 'data',
        '--num-samples', '1',
        '--cpus', '1',
        '--gpus', '0',
        '--trial-cpus', '1',
        '--trial-gpus', '0'))

    variant_spec = module.get_variant_spec(example_args)
    kwargs = generate_experiment_kwargs(variant_spec, example_args)

    kwargs['config'][
        'environment_params'][
        'training'][
        'kwargs'][
        'fixed_design'] = DEFAULT_DESIGN
    kwargs['config'][
        'environment_params'][
        'training'][
        'kwargs'][
        'expose_design'] = False

    env_params = kwargs['config']['environment_params']
    env = get_environment_from_params(env_params['training'])

    kwargs['config']['policy_params']['config'].update({
        'action_range': (env.action_space.low, env.action_space.high),
        'input_shapes': env.observation_shape,
        'output_shape': env.action_shape})

    policy = policies.get(kwargs['config']['policy_params'])
    policy.load_weights(
        args.ckpt).assert_consumed().run_restore_ops()

    weights = policy.get_weights()
    with open('dkitty_oracle.pkl', 'wb') as f:
        pkl.dump(weights, f)
