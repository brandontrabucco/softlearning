from softlearning.environments.gym.mujoco.morphing_dkitty import DEFAULT_DKITTY
from softlearning.environments.gym.mujoco.morphing_dkitty import UPPER_BOUND
from softlearning.environments.gym.mujoco.morphing_dkitty import LOWER_BOUND
from softlearning.environments.gym.mujoco.morphing_dkitty import Leg
from examples.instrument import generate_experiment_kwargs
from ray import tune
from math import floor


import numpy as np
import argparse
import importlib
import ray
import multiprocessing
import tensorflow as tf


if __name__ == '__main__':

    parser = argparse.ArgumentParser('MorphingDKitty')
    parser.add_argument('--num_legs', type=int, default=4)
    parser.add_argument('--dataset_size', type=int, default=100)
    parser.add_argument('--num_parallel', type=int, default=12)
    args = parser.parse_args()

    LEGS_SPEC = []
    for i in range(args.dataset_size):
        if i == 0 and len(DEFAULT_DKITTY) == args.num_legs:
            LEGS_SPEC.append(DEFAULT_DKITTY)
        else:
            LEGS_SPEC.append([Leg(*np.random.uniform(
                LOWER_BOUND, UPPER_BOUND)) for _ in range(args.num_legs)])

    def run_example(example_module_name, example_argv, local_mode=False):
        """Run example locally, potentially parallelizing across cpus/gpus."""
        example_module = importlib.import_module(example_module_name)

        example_args = example_module.get_parser().parse_args(example_argv)
        variant_spec = example_module.get_variant_spec(example_args)
        trainable_class = example_module.get_trainable_class(example_args)

        experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)
        experiment_kwargs['config'][
            'dataset_id'] = tune.grid_search(list(range(args.dataset_size)))

        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'legs'] = tune.sample_from(
                lambda spec: LEGS_SPEC[spec.config.dataset_id])

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
        '--domain', 'MorphingDKitty',
        '--task', 'v0',
        '--exp-name', f'morphing-dkitty',
        '--checkpoint-frequency', '10',
        '--mode=local',
        '--num-samples', '3'
        '--cpus', f'{num_cpus}',
        '--gpus', f'{num_gpus}',
        '--trial-cpus', f'{num_cpus // args.num_parallel}',
        '--trial-gpus', f'{floor(num_gpus/args.num_parallel/0.1)*0.1}'))
