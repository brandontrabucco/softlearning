from morphing_agents.mujoco.dkitty.designs import sample_uniformly
from morphing_agents.mujoco.dkitty.designs import DEFAULT_DESIGN
from morphing_agents.mujoco.dkitty.elements import LEG_UPPER_BOUND
from morphing_agents.mujoco.dkitty.elements import LEG_LOWER_BOUND
from morphing_agents.mujoco.dkitty.elements import LEG
from morphing_agents.mujoco.dkitty.env import MorphingDKittyEnv
from examples.instrument import generate_experiment_kwargs
from examples.development.variants import TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK
from ray import tune
from math import floor
import argparse
import importlib
import ray
import multiprocessing
import tensorflow as tf
import numpy as np


if __name__ == '__main__':

    parser = argparse.ArgumentParser('GenerateDKittyDataset')
    parser.add_argument('--local-dir',
                        type=str,
                        default='./data')
    parser.add_argument('--num-legs',
                        type=int,
                        default=4)
    parser.add_argument('--dataset-size',
                        type=int,
                        default=1)
    parser.add_argument('--num-samples',
                        type=int,
                        default=1)
    parser.add_argument('--num-parallel',
                        type=int,
                        default=1)
    parser.add_argument('--method',
                        type=str,
                        choices=['uniform', 'curated'])
    args = parser.parse_args()

    TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK[
        'gym']['MorphingDKitty']['v0'] = 100000

    ub = np.array(list(LEG_UPPER_BOUND))
    lb = np.array(list(LEG_LOWER_BOUND))
    scale = (ub - lb) / 2

    designs = [DEFAULT_DESIGN]
    while len(designs) < args.dataset_size:
        try:

            if args.method == 'uniform':
                d = sample_uniformly(num_legs=args.num_legs)
            elif args.method == 'curated':
                d = [LEG(*np.clip(np.array(
                    leg) + np.random.normal(0, scale / 8), lb, ub))
                    for leg in DEFAULT_DESIGN]
            else:
                d = DEFAULT_DESIGN

            MorphingDKittyEnv(fixed_design=d)
            designs.append(d)
        except Exception:
            print(f"resampling design that errored: {d}")

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
            'fixed_design'] = tune.sample_from(
                lambda spec: designs[spec.config.dataset_id])
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
        '--domain', 'MorphingDKitty',
        '--task', 'v0',
        '--exp-name', f'dkitty-dataset-{args.method}',
        '--checkpoint-frequency', '10',
        '--mode=local',
        '--local-dir', args.local_dir,
        '--num-samples', f'{args.num_samples}',
        '--cpus', f'{num_cpus}',
        '--gpus', f'{num_gpus}',
        '--trial-cpus', f'{num_cpus // args.num_parallel}',
        '--trial-gpus', f'{num_gpus / args.num_parallel}'))
