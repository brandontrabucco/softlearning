from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN
from examples.instrument import generate_experiment_kwargs as gen_kwargs
from examples.development.variants import TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK
from ray import tune
import argparse
import importlib
import ray
import multiprocessing
import tensorflow as tf


if __name__ == '__main__':

    parser = argparse.ArgumentParser('AntOracle')
    parser.add_argument('--local-dir', type=str, default='./ant-oracle')
    args = parser.parse_args()

    TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK[
        'gym']['MorphingAnt']['v0'] = 3000000

    def run_example(mod_name, example_argv, local_mode=False):
        mod = importlib.import_module(mod_name)
        example_args = mod.get_parser().parse_args(example_argv)
        variant_spec = mod.get_variant_spec(example_args)

        experiment_kwargs = gen_kwargs(variant_spec, example_args)
        trainable_class = mod.get_trainable_class(example_args)

        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'fixed_design'] = DEFAULT_DESIGN
        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'expose_design'] = False

        ray.init(
            num_cpus=example_args.cpus,
            num_gpus=example_args.gpus,
            resources=example_args.resources or {},
            local_mode=local_mode,
            include_webui=example_args.include_webui,
            temp_dir=example_args.temp_dir)

        tune.run(
            trainable_class,
            **experiment_kwargs,
            with_server=example_args.with_server,
            server_port=example_args.server_port,
            scheduler=None,
            reuse_actors=True)

    num_cpus = multiprocessing.cpu_count()
    num_gpus = len(tf.config.list_physical_devices('GPU'))
    run_example('examples.development', (
        '--algorithm', 'SAC',
        '--universe', 'gym',
        '--domain', 'MorphingAnt',
        '--task', 'v0',
        '--exp-name', f'ant-oracle',
        '--checkpoint-frequency', '10',
        '--mode=local',
        '--local-dir', args.local_dir,
        '--num-samples', '1',
        '--cpus', f'{num_cpus}',
        '--gpus', f'{num_gpus}',
        '--trial-cpus', f'{num_cpus}',
        '--trial-gpus', f'{num_gpus}'))
