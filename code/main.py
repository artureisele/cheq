import argparse

import warnings
from copy import deepcopy

from utils.tools import get_kwargs

from configs import RACING, CHEQ_ENTRY_POINT, RL_BASELINE_ENTRY_POINT, \
    C_CORE_ENTRY_POINT, CORE_ENTRY_POINT, REDQ_BASELINE_ENTRY_POINT, CARTPOLE

warnings.filterwarnings("ignore", category=DeprecationWarning)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('-algo', '--algorithm', default="cheq", help='The algorithm you want to run (cheq, core, c-core, bcf, c-bcf, rl, redq)')
    parser.add_argument('-rname', '--run-name', help='The name of the run. Defaults to the algorithm name.')
    parser.add_argument('-train-start', '--train-start', type=int, default=1000, help='When the training begins.')
    parser.add_argument('-start-steps', '--start_steps', type=int, default=5000, help='When the algorithm starts completely.')
    parser.add_argument('-exp', '--exp-type', help='exp-type for better structuring of the experiments. Defaults to the env.')
    parser.add_argument('-env', '--environment', default='racing', help='The environment you want to run (racing, cartpole)')
    parser.add_argument('-steps', '--training-steps', type=int, default=int(1.5e6), help='The number of training steps.')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='The training run seed.')
    parser.add_argument('-ens-size', '--ensemble-size', type=int,  default=5, help='The size of the ensemble used (res-ada, bcf, redq)')
    parser.add_argument('-sig-prior', '--sigma-prior', type=float, default=0.2, help='Sigma Prior for (C-)BCF.')
    parser.add_argument('-agg', '--weight-aggregate', default='mean', help='which aggregate to use for the weight in (C-)BCF. (mean, min)')
    parser.add_argument('-a', '--lambda-max', type=float, default=7.0, help='Lambda Max for (C-)CORE.')
    parser.add_argument('-c', '--factor-c', type=float, default=0.3, help='The factor c for (C-)CORE.')
    parser.add_argument('-uhigh', '--u-high', type=float, default=0.15, help='Uncertainty High for CHEQ')
    parser.add_argument('-ulow', '--u-low', type=float, default=0.03, help='Uncertainty High for CHEQ')
    parser.add_argument('-lam-high', '--lambda-high', type=float, default=1, help='Lambda high for CHEQ')
    parser.add_argument('-lam-low', '--lambda-low', type=float, default=0.2, help='Lambda low for CHEQ')
    parser.add_argument('-lam-warm', '--lambda-warmup', type=float, default=0.3, help='Warmup lambda for C-variants.')
    parser.add_argument('-pfreq', '--policy-frequency', type=int, default=2, help='The policy update frequency.')
    parser.add_argument('-G', '--update-steps', type=int, default=20, help='Number of Q-updates per environment step (REDQ, CHEQ).')

    args = parser.parse_args()

    if args.exp_type is None:
        args.exp_type = args.environment

    if args.run_name is None:
        args.run_name = args.algorithm

    if args.algorithm in ['cheq', 'core', 'c-core', 'rl', 'redq']:
        # Choose the right Entry Point
        if args.algorithm == 'cheq':
            ENTRY_POINT = deepcopy(CHEQ_ENTRY_POINT)
            entry_point = ENTRY_POINT['entry_point']['type']
            entry_point_kwargs = get_kwargs(ENTRY_POINT['entry_point'])
        elif args.algorithm == 'core':
            ENTRY_POINT = deepcopy(CORE_ENTRY_POINT)
            entry_point = ENTRY_POINT['entry_point']['type']
            entry_point_kwargs = get_kwargs(ENTRY_POINT['entry_point'])
        elif args.algorithm == 'c-core':
            ENTRY_POINT = deepcopy(C_CORE_ENTRY_POINT)
            entry_point = ENTRY_POINT['entry_point']['type']
            entry_point_kwargs = get_kwargs(ENTRY_POINT['entry_point'])
        elif args.algorithm == 'rl':
            ENTRY_POINT = deepcopy(RL_BASELINE_ENTRY_POINT)
            entry_point = ENTRY_POINT['entry_point']['type']
            entry_point_kwargs = get_kwargs(ENTRY_POINT['entry_point'])
        elif args.algorithm == 'redq':
            ENTRY_POINT = deepcopy(REDQ_BASELINE_ENTRY_POINT)
            entry_point = ENTRY_POINT['entry_point']['type']
            entry_point_kwargs = get_kwargs(ENTRY_POINT['entry_point'])

        # Update name and log folder
        ENTRY_POINT['run_name'] = args.run_name
        ENTRY_POINT['exp_type'] = args.exp_type

        # Update environment
        if args.environment == 'racing':
            ENTRY_POINT['env'] = deepcopy(RACING)
        elif args.environment == 'cartpole':
            ENTRY_POINT['env'] = deepcopy(CARTPOLE)
        else:
            raise NotImplementedError()

        # Update general hyperparams
        ENTRY_POINT['training']['number_steps'] = args.training_steps
        ENTRY_POINT['training']['seed'] = args.seed
        ENTRY_POINT['training']['agent_trainer']['learning_starts'] = args.train_start
        ENTRY_POINT['training']['agent_trainer']['policy_frequency'] = args.policy_frequency
        ENTRY_POINT['training']['agent_trainer']['update_steps'] = args.update_steps

        # setting algorithm specific hyperparams
        if args.algorithm in ['cheq', 'core', 'c-core']:
            ENTRY_POINT['training']['weight_scheduler']['t_start'] = args.start_steps

        if args.algorithm == 'cheq':
            ENTRY_POINT['training']['agent_trainer']['ensemble_size'] = args.ensemble_size
            ENTRY_POINT['training']['weight_scheduler']['u_high'] = args.u_high
            ENTRY_POINT['training']['weight_scheduler']['u_low'] = args.u_low
            ENTRY_POINT['training']['weight_scheduler']['lambda_max'] = args.lambda_high
            ENTRY_POINT['training']['weight_scheduler']['lambda_min'] = args.lambda_low
            ENTRY_POINT['training']['weight_scheduler']['lambda_warmup_max'] = args.lambda_warmup
        elif args.algorithm in ['core', 'c-core']:
            ENTRY_POINT['training']['weight_scheduler']['factor_c'] = args.factor_c
            ENTRY_POINT['training']['weight_scheduler']['factor_a'] = args.lambda_max
            if args.algorithm == 'c-core':
                ENTRY_POINT['training']['weight_scheduler']['lambda_warmup_max'] = args.lambda_warmup
        elif args.algorithm == 'redq':
            ENTRY_POINT['training']['agent_trainer']['ensemble_size'] = args.ensemble_size

        entry_point(config=ENTRY_POINT, **entry_point_kwargs)

    elif args.algorithm == 'bcf' or args.algorithm == 'c-bcf':
        import baselines.bcf_main.main as bcf
        bcf.METHOD = 'BCF'
        bcf.NUM_AGENTS = args.ensemble_size
        bcf.SEED = args.seed
        bcf.NUM_STEPS = args.training_steps
        bcf.TRAINING_START = args.train_start
        bcf.START_STEPS = args.start_steps
        bcf.SIGMA_PRIOR = args.sigma_prior
        bcf.lambda_context = True if args.algorithm == 'c-bcf' else False
        bcf.sigma_prior = bcf.SIGMA_PRIOR
        bcf.POLICY_FREQUENCY = args.policy_frequency
        bcf.WEIGHT_AGGREGATE = args.weight_aggregate
        bcf.WARMUP_WEIGHT = args.lambda_warmup

        if args.environment == 'racing':
            bcf.TASK = 'racing'
            bcf.ENV = 'CarRacingEnv'
            bcf.NUM_TEST_EPISODES = 1

            from baselines.bcf_main.car_racing.car_racing_env import CarRacingEnv
            from baselines.bcf_main.car_racing.prior_controller import PathFollowingController

            bcf.max_ep_steps = 600
            bcf.env = CarRacingEnv()
            bcf.prior = PathFollowingController(bcf.env)

        else:
            raise NotImplementedError()

        bcf.update_hypers()
        bcf.init_run(args.run_name, args.exp_type)
        bcf.run(bcf.agents, bcf.env)
    else:
        raise NotImplementedError()