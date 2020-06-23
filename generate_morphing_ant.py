import numpy as np
import argparse
from softlearning.environments.gym.mujoco.morphing_ant import DEFAULT_ANT
from softlearning.environments.gym.mujoco.morphing_ant import UPPER_BOUND
from softlearning.environments.gym.mujoco.morphing_ant import LOWER_BOUND
from softlearning.environments.gym.mujoco.morphing_ant import Leg
from examples.instrument import run_example_local
from examples.instrument import generate_experiment_kwargs


import importlib
import ray
from ray import tune


if __name__ == '__main__':

    parser = argparse.ArgumentParser('MorphingAnt')
    parser.add_argument('--num_legs', type=int, default=4)
    parser.add_argument('--dataset_size', type=int, default=5)
    parser.add_argument('--num_parallel', type=int, default=5)
    args = parser.parse_args()

    def run_example(example_module_name, example_argv, local_mode=False):
        """Run example locally, potentially parallelizing across cpus/gpus."""
        example_module = importlib.import_module(example_module_name)

        example_args = example_module.get_parser().parse_args(example_argv)
        variant_spec = example_module.get_variant_spec(example_args)
        trainable_class = example_module.get_trainable_class(example_args)

        experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)

        dataset_id = list(range(args.dataset_size))

        legs_spec = []
        for i in dataset_id:
            if i == 0 and len(DEFAULT_ANT) == args.num_legs:
                legs_spec.append(DEFAULT_ANT)
            else:
                legs_spec.append([Leg(*np.random.uniform(
                    LOWER_BOUND, UPPER_BOUND)) for _ in range(args.num_legs)])

        experiment_kwargs['config']['dataset_id'] = tune.grid_search(dataset_id)

        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'legs'] = tune.sample_from(
                lambda spec: legs_spec[spec.config.dataset_id])

        experiment_kwargs['config'][
            'environment_params'][
            'evaluation'][
            'kwargs'][
            'legs'] = tune.sample_from(
                lambda spec: legs_spec[spec.config.dataset_id])

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

    run_example_local('examples.development', (
        '--algorithm', 'SAC',
        '--universe', 'gym',
        '--domain', 'MorphingAnt',
        '--task', 'v0',
        '--exp-name', f'morphing-ant',
        '--checkpoint-frequency', '1000',
        '--mode=local',
        '--cpus', '24',
        '--gpus', '2',
        '--trial-cpus', '2',
        '--trial-gpus', '0.166'))
