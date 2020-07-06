from examples.instrument import generate_experiment_kwargs
from examples.development.variants import TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK
from ray import tune
from math import floor
import pickle as pkl
import os
import argparse
import importlib
import ray
import multiprocessing
import tensorflow as tf


if __name__ == '__main__':

    parser = argparse.ArgumentParser('EvaluateAntDesign')
    parser.add_argument('--local-dir',
                        type=str,
                        default='./data')
    parser.add_argument('--designs',
                        type=str,
                        default='designs.pkl')
    parser.add_argument('--num-samples',
                        type=int,
                        default=1)
    parser.add_argument('--num-parallel',
                        type=int,
                        default=1)
    args = parser.parse_args()

    TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK[
        'gym']['MorphingAnt']['v0'] = 3000000

    with open(args.designs, "rb") as f:
        designs = pkl.load(f)
        print(designs)
        dataset_size = len(designs)

    def run_example(example_module_name, example_argv, local_mode=False):
        """Run example locally, potentially parallelizing across cpus/gpus."""
        example_module = importlib.import_module(example_module_name)

        example_args = example_module.get_parser().parse_args(example_argv)
        variant_spec = example_module.get_variant_spec(example_args)
        trainable_class = example_module.get_trainable_class(example_args)

        experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)
        experiment_kwargs['config'][
            'dataset_id'] = tune.grid_search(list(range(dataset_size)))

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
        '--domain', 'MorphingAnt',
        '--task', 'v0',
        '--exp-name', f'{os.path.basename(args.designs).replace(".pkl", "")}',
        '--checkpoint-frequency', '10',
        '--mode=local',
        '--local-dir', args.local_dir,
        '--num-samples', f'{args.num_samples}',
        '--cpus', f'{num_cpus}',
        '--gpus', f'{num_gpus}',
        '--trial-cpus', f'{num_cpus // args.num_parallel}',
        '--trial-gpus', f'{floor(num_gpus/args.num_parallel/0.1)*0.1}'))
