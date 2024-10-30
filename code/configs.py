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

CARTPOLE_TRAIN_CONFIG = deepcopy(TRAIN_CONFIG)
CARTPOLE_TRAIN_CONFIG['agent_trainer']['hidden_layer_size_q'] = 64
CARTPOLE_TRAIN_CONFIG['agent_trainer']['hidden_layer_size_actor'] = 16

RACING_TRAIN_CONFIG = deepcopy(TRAIN_CONFIG)
RACING_TRAIN_CONFIG['agent_trainer']['q_lr'] = 3e-4
RACING_TRAIN_CONFIG['agent_trainer']['learning_starts'] = 1000

NAVIGATION_TRAIN_CONFIG = deepcopy(TRAIN_CONFIG)
NAVIGATION_TRAIN_CONFIG['agent_trainer']['q_lr'] = 3e-4
NAVIGATION_TRAIN_CONFIG['number_steps'] = int(1e6)

CARTPOLE = {
    'env_id': 'CustomCartPole-v1',
    'make_env_function': {'type': make_env, 'env_id': 'CustomCartPole-v1'},
    'evaluator': {'type': CartPoleEvaluator},
    'controller': {'type': CartPoleController, 'K': [[-2.02814789, -3.48532665, -20.0614485, -8.58323148]]},
    'observation_manager': {'type': CartPoleObservationManager}
}

RACING = {
    'env_id': 'CarEnv-v1',
    'make_env_function': {'type': make_racing_env, 'no_training_tracks': 1,
    #                      'training_tracks_path': 'custom_envs/CarEnv/SavedTracks/10_test_tracks'},
                          'training_tracks_path': 'custom_envs/CarEnv/SavedTracks/training_track'},
    'evaluator': {'type': CarRacingEvaluator},
    'controller': {'type': LinearGainScheduleRacingController,
                   "k_angle": 1.0, "k_cross_track": -0.5, "k_velocity": 0.25, "k_curvature_rad": 0.4,
                   "max_target_vel": 8, "vel_low": 8, "vel_high": 28, "gain_low": 0.2},
    'observation_manager': {'type': CarRacingObservationManager}
}

ENTRY_POINT_ABSTRACT = {
    'env': RACING,
    'training': TRAIN_CONFIG,
    'evaluation': EVALUATION,
    'logging': LOGGING,
}

# RL BASELINE CONFIG
RL_BASELINE_ENTRY_POINT = deepcopy(ENTRY_POINT_ABSTRACT)
RL_BASELINE_ENTRY_POINT['training'] = deepcopy(RACING_TRAIN_CONFIG)
RL_BASELINE_ENTRY_POINT['entry_point'] = {'type': train_rl}

# REDQ BASELINE CONFIG
REDQ_BASELINE_ENTRY_POINT = deepcopy(ENTRY_POINT_ABSTRACT)
REDQ_BASELINE_ENTRY_POINT['training'] = deepcopy(RACING_TRAIN_CONFIG)
REDQ_BASELINE_ENTRY_POINT['training']['agent_trainer']['ensemble_size'] = 5
REDQ_BASELINE_ENTRY_POINT['training']['agent_trainer']['update_steps'] = 20
REDQ_BASELINE_ENTRY_POINT['training']['agent_trainer']['pi_update_avg_q'] = True
REDQ_BASELINE_ENTRY_POINT['entry_point'] = {'type': train_rl}

# CHEQ CONFIG
CHEQ_ENTRY_POINT = deepcopy(ENTRY_POINT_ABSTRACT)
CHEQ_ENTRY_POINT['training'] = deepcopy(RACING_TRAIN_CONFIG)
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


# C-CORE CONFIG
C_CORE_ENTRY_POINT = deepcopy(ENTRY_POINT_ABSTRACT)
C_CORE_ENTRY_POINT['training'] = deepcopy(RACING_TRAIN_CONFIG)
C_CORE_ENTRY_POINT['training']['weight_scheduler'] = {'type': CoreExponentialWeightScheduler,
                                                                'factor_a': 7, 'factor_c': 0.4, 't_start': 5000,
                                                                'lambda_warmup': 0.2, 'lambda_warmup_max': 0.3}
C_CORE_ENTRY_POINT['training']['uncertainty_evaluator'] = {'type': TargetTDErrorUncertaintyEvaluator, 'gamma': 0.99}
C_CORE_ENTRY_POINT['entry_point'] = {'type': train_apr}


# CORE CONFIG
CORE_ENTRY_POINT = deepcopy(ENTRY_POINT_ABSTRACT)
CORE_ENTRY_POINT['training'] = deepcopy(RACING_TRAIN_CONFIG)
CORE_ENTRY_POINT['training']['weight_scheduler'] = {'type': CoreExponentialWeightScheduler,
                                                                'factor_a': 7, 'factor_c': 0.4, 't_start': 5000,
                                                                'lambda_warmup': 0.2, 'lambda_warmup_max': None}
CORE_ENTRY_POINT['training']['uncertainty_evaluator'] = {'type': TargetTDErrorUncertaintyEvaluator, 'gamma': 0.99}
CORE_ENTRY_POINT['entry_point'] = {'type': train_ada_rl}

