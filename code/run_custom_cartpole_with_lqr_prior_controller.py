from copy import deepcopy

from custom_envs.observation_manager import CarRacingObservationManager, CartPoleObservationManager
from utils.tools import make_racing_env, make_env, \
    CoreExponentialWeightScheduler, DummyUncertaintyEvaluator, FixedWeightScheduler
from utils.evaluator import CartPoleEvaluator, CarRacingEvaluator
from custom_envs.controllers import CartPoleController, LinearGainScheduleRacingController
from simple_rl_baseline import train as train_rl
from hybrid_rl import train as train_ada_rl
from contextualized_hybrid_rl import train as train_apr
from sac_trainer import SACTrainer

from utils.tools import QEnsembleSTDUncertaintyEvaluator, MovingAVGLinearWeightScheduler, TargetTDErrorUncertaintyEvaluator

import argparse
import warnings
from copy import deepcopy

from utils.tools import get_kwargs
from configs import CARTPOLE, REDQ_BASELINE_ENTRY_POINT

warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-algo', '--algorithm', default="redq", help='The algorithm you want to run (cheq, core, c-core, bcf, c-bcf, rl, redq)')
    parser.add_argument('-rname', '--run-name', help='The name of the run. Defaults to the algorithm name.')
    parser.add_argument('-train-start', '--train-start', type=int, default=1000, help='When the training begins.')
    parser.add_argument('-start-steps', '--start_steps', type=int, default=5000, help='When the algorithm starts completely.')
    parser.add_argument('-exp', '--exp-type', help='exp-type for better structuring of the experiments. Defaults to the env.')
    parser.add_argument('-env', '--environment', default='cartpole', help='The environment you want to run (racing, cartpole)')
    parser.add_argument('-steps', '--training-steps', type=int, default=int(1.5e6), help='The number of training steps.')
    parser.add_argument('-seed', '--seed', type=int, default=0, help='The training run seed.')
    parser.add_argument('-ens-size', '--ensemble-size', type=int,  default=5, help='The size of the ensemble used (res-ada, bcf, redq)')
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
    print(args)
    LOGGING = {
        'wandb_project_name': 'CHEQ-RLC-2024',
        'capture_video': True,
        'model_save_frequency': 5000
    }

    EVALUATION = {
        'eval_agent': True,
        'eval_count': 10,
        'eval_frequency': 20,
    }

    TRAIN_CONFIG = {
        'agent_trainer': {'type': SACTrainer, 'hidden_layer_size_q': 265, 'hidden_layer_size_actor': 256},
        'number_steps': int(1.5e6),
        'device': 'cuda:0',
        'seed': 1,
    }
    CARTPOLE = {
        'env_id': 'CustomCartPole-v1',
        'make_env_function': {'type': make_env, 'env_id': 'CustomCartPole-v1'},
        'evaluator': {'type': CartPoleEvaluator},
        'controller': {'type': CartPoleController},
        'observation_manager': {'type': CartPoleObservationManager}
    }
    ENTRY_POINT_ABSTRACT = {
        'env': CARTPOLE,
        'training': TRAIN_CONFIG,
        'evaluation': EVALUATION,
        'logging': LOGGING,
    }

    CARTPOLE_TRAIN_CONFIG = deepcopy(TRAIN_CONFIG)
    CARTPOLE_TRAIN_CONFIG['agent_trainer']['hidden_layer_size_q'] = 64
    CARTPOLE_TRAIN_CONFIG['agent_trainer']['hidden_layer_size_actor'] = 16
    # Set up entry point for REDQ and CHEQ configuration
    CHEQ_ENTRY_POINT = deepcopy(ENTRY_POINT_ABSTRACT)
    CHEQ_ENTRY_POINT['training'] = deepcopy(CARTPOLE_TRAIN_CONFIG)
    CHEQ_ENTRY_POINT['training']['agent_trainer']['ensemble_size'] = 3
    CHEQ_ENTRY_POINT['training']['agent_trainer']['bernoulli_mask_coeff'] = 0.8
    CHEQ_ENTRY_POINT['training']['agent_trainer']['use_rpf'] = False
    CHEQ_ENTRY_POINT['training']['agent_trainer']['pi_update_avg_q'] = True
    CHEQ_ENTRY_POINT['training']['weight_scheduler'] = {'type': MovingAVGLinearWeightScheduler,
                                                                    'lambda_min': 0.2, 'lambda_max': 1.0, 't_start': 5000,
                                                                    'u_high': 0.15, 'u_low': 0.03, 'window_size': 1,
                                                                    'lambda_warmup_max': 0.3}
    CHEQ_ENTRY_POINT['training']['uncertainty_evaluator'] = {'type': QEnsembleSTDUncertaintyEvaluator}
    CHEQ_ENTRY_POINT['entry_point'] = {'type': train_apr}

    # Run the configured entry point
    CHEQ_ENTRY_POINT['run_name'] = args.run_name
    CHEQ_ENTRY_POINT['exp_type'] = args.exp_type
    CHEQ_ENTRY_POINT['training']['number_steps'] = args.training_steps
    CHEQ_ENTRY_POINT['training']['seed'] = args.seed
    CHEQ_ENTRY_POINT['training']['agent_trainer']['learning_starts'] = args.train_start
    CHEQ_ENTRY_POINT['training']['agent_trainer']['policy_frequency'] = args.policy_frequency
    CHEQ_ENTRY_POINT['training']['agent_trainer']['update_steps'] = args.update_steps
    CHEQ_ENTRY_POINT['training']['agent_trainer']['ensemble_size'] = args.ensemble_size
    CHEQ_ENTRY_POINT['training']['weight_scheduler']['u_high'] = args.u_high
    CHEQ_ENTRY_POINT['training']['weight_scheduler']['u_low'] = args.u_low
    CHEQ_ENTRY_POINT['training']['weight_scheduler']['lambda_max'] = args.lambda_high
    CHEQ_ENTRY_POINT['training']['weight_scheduler']['lambda_min'] = args.lambda_low
    CHEQ_ENTRY_POINT['training']['weight_scheduler']['lambda_warmup_max'] = args.lambda_warmup
    
    entry_point = CHEQ_ENTRY_POINT['entry_point']['type']
    entry_point_kwargs = get_kwargs(CHEQ_ENTRY_POINT['entry_point'])
    entry_point(config=CHEQ_ENTRY_POINT, **entry_point_kwargs)

if __name__ == '__main__':
    main()
