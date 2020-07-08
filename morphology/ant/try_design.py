from examples.instrument import generate_experiment_kwargs
from examples.development.variants import TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK
from morphing_agents.mujoco.ant.elements import LEG
from ray import tune
import argparse
import importlib
import ray
import multiprocessing
import tensorflow as tf


BAD_DESIGN = [
    LEG(x=-0.01752557260248648,
        y=-0.044383452339044505,
        z=0.035900934167950344,
        a=-176.4195801607484,
        b=116.68176744690157,
        c=35.268085280820344,
        hip_upper=11.19603545382594,
        hip_lower=-49.46392195321905,
        thigh_upper=51.0657131726093,
        thigh_lower=-14.615106012390918,
        ankle_upper=74.57176333327824,
        ankle_lower=56.473702804353934,
        hip_size=0.29823093428061825,
        thigh_size=0.29646611375533083,
        ankle_size=0.6441347845016865),
    LEG(x=0.03496779544477463,
        y=0.07108504122970247,
        z=-0.03569704198856287,
        a=55.64648224655676,
        b=34.441818144624534,
        c=128.52145488754383,
        hip_upper=2.765140451501311,
        hip_lower=-7.805535509168628,
        thigh_upper=16.581956331053753,
        thigh_lower=-16.643597452043643,
        ankle_upper=118.82753177520526,
        ankle_lower=17.41881784645519,
        hip_size=0.34530722883833775,
        thigh_size=0.23448506603466768,
        ankle_size=0.6420048202696035),
    LEG(x=-0.032470586985765534,
        y=-0.04473449397619514,
        z=-0.04842402058229145,
        a=-147.23479919574032,
        b=-89.35407443045268,
        c=-6.695254247948668,
        hip_upper=39.55393812599064,
        hip_lower=-53.067012630493096,
        thigh_upper=57.65507028195125,
        thigh_lower=-16.087327083146604,
        ankle_upper=84.64298937095785,
        ankle_lower=56.113804601571054,
        hip_size=0.22588675560970153,
        thigh_size=0.286648206672234,
        ankle_size=0.5973884071392149),
    LEG(x=0.05334992704907848,
        y=-0.0879241340931864,
        z=0.004781725263273515,
        a=-160.96831368893146,
        b=-46.40782597864799,
        c=-144.58495413419965,
        hip_upper=7.979335068756887,
        hip_lower=-33.96012769698596,
        thigh_upper=21.344848783389082,
        thigh_lower=-39.90798896053889,
        ankle_upper=118.97687622111908,
        ankle_lower=26.683489542854673,
        hip_size=0.1588388652402842,
        thigh_size=0.18433120777152506,
        ankle_size=0.5539785875000842)]


if __name__ == '__main__':

    parser = argparse.ArgumentParser('EvaluateAntDesign')
    parser.add_argument('--local-dir',
                        type=str,
                        default='./data')
    parser.add_argument('--num-samples',
                        type=int,
                        default=1)
    parser.add_argument('--num-parallel',
                        type=int,
                        default=1)
    args = parser.parse_args()

    TOTAL_STEPS_PER_UNIVERSE_DOMAIN_TASK[
        'gym']['MorphingAnt']['v0'] = 3000000

    def run_example(example_module_name, example_argv, local_mode=False):
        """Run example locally, potentially parallelizing across cpus/gpus."""
        example_module = importlib.import_module(example_module_name)

        example_args = example_module.get_parser().parse_args(example_argv)
        variant_spec = example_module.get_variant_spec(example_args)
        trainable_class = example_module.get_trainable_class(example_args)

        experiment_kwargs = generate_experiment_kwargs(variant_spec, example_args)

        experiment_kwargs['config'][
            'environment_params'][
            'training'][
            'kwargs'][
            'fixed_design'] = BAD_DESIGN
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
        '--exp-name', f'bad-design',
        '--checkpoint-frequency', '10',
        '--mode=local',
        '--local-dir', args.local_dir,
        '--num-samples', f'{args.num_samples}',
        '--cpus', f'{num_cpus}',
        '--gpus', f'{num_gpus}',
        '--server-port', '9032',
        '--trial-cpus', f'{num_cpus // args.num_parallel}',
        '--trial-gpus', f'{num_gpus / args.num_parallel}'))
