from examples.instrument import generate_experiment_kwargs
from examples.development.variants import TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK
from morphing_agents.mujoco.ant.designs import DEFAULT_DESIGN
from copy import deepcopy
from ray import tune
import argparse
import importlib
import ray
import multiprocessing
import tensorflow as tf


if __name__ == '__main__':

    parser = argparse.ArgumentParser('TrainAntOracle')
    parser.add_argument('--local-dir',
                        type=str,
                        default='./data')
    parser.add_argument('--num-legs',
                        type=int,
                        default=4)
    parser.add_argument('--num-samples',
                        type=int,
                        default=1)
    parser.add_argument('--num-parallel',
                        type=int,
                        default=1)
    args = parser.parse_args()

    TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK[
        'gym']['MorphingAnt']['v0'] = 100000000

    def run_example(example_module_name, example_argv, local_mode=False):
        """Run example locally, potentially parallelizing across cpus/gpus."""
        example_module = importlib.import_module(example_module_name)

        example_args = example_module.get_parser().parse_args(example_argv)
        variant_spec = example_module.get_variant_spec(example_args)
        trainable_class = example_module.get_trainable_class(example_args)

        experiment_kwargs = generate_experiment_kwargs(variant_spec,
                                                       example_args)

        # Training environment parameters
        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'num_legs'] = args.num_legs
        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'fixed_design'] = None
        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'expose_design'] = True

        # Evaluation environment parameters
        experiment_kwargs['config'][
            'environment_params'][
            'evaluation'] = deepcopy(
                experiment_kwargs['config'][
                    'environment_params'][
                    'training'])
        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'fixed_design'] = DEFAULT_DESIGN
        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'expose_design'] = True

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
        '--exp-name', 'ant-oracle',
        '--checkpoint-frequency', '10',
        '--mode=local',
        '--local-dir', args.local_dir,
        '--num-samples', f'{args.num_samples}',
        '--cpus', f'{num_cpus}',
        '--gpus', f'{num_gpus}',
        '--trial-cpus', f'{num_cpus // args.num_parallel}',
        '--trial-gpus', f'{num_gpus / args.num_parallel}'))
