from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN
from examples.instrument import generate_experiment_kwargs
from softlearning.environments.utils import get_environment_from_params
from softlearning import policies
import argparse
import importlib


if __name__ == '__main__':

    parser = argparse.ArgumentParser('AntOracle')
    parser.add_argument('--ckpt', type=str, default='./ant-oracle')
    args = parser.parse_args()

    module = importlib.import_module('examples.development')
    example_args = module.get_parser().parse_args((
        '--algorithm', 'SAC',
        '--universe', 'gym',
        '--domain', 'MorphingAnt',
        '--task', 'v0',
        '--exp-name', f'ant-oracle',
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

    obs = env.reset()
    reward = 0.0
    d = None

    for i in range(1000):

        action = policy.action(obs).numpy()
        obs, rt, d, info = env.step(action)
        reward += rt

        env.render(mode='human')

        if d:
            break
