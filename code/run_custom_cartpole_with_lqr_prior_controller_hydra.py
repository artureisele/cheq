from omegaconf import DictConfig, OmegaConf
import hydra
from custom_envs.observation_manager import CarRacingObservationManager, CartPoleObservationManager
from utils.tools import make_racing_env, make_env, \
    CoreExponentialWeightScheduler, DummyUncertaintyEvaluator, FixedWeightScheduler
from utils.evaluator import CartPoleEvaluator, CarRacingEvaluator
from custom_envs.controllers import CartPoleController, LinearGainScheduleRacingController
from simple_rl_baseline import train as train_rl
from hybrid_rl import train as train_ada_rl
from contextualized_hybrid_rl import train as train_apr
from categorical_ensemble_contextualized_hybrid_rl import train as categorical_train_apr
from sac_trainer import SACTrainer, CategoricalSACTrainer

from utils.tools import QEnsembleSTDUncertaintyEvaluator, MovingAVGLinearWeightScheduler, TargetTDErrorUncertaintyEvaluator


def fill_cfg_with_functions(cfg: DictConfig)-> DictConfig:
    cfg = OmegaConf.to_container(cfg, resolve=True)
    cfg["training"]["weight_scheduler"]["type"] = MovingAVGLinearWeightScheduler
    cfg["training"]["uncertainty_evaluator"]["type"]=QEnsembleSTDUncertaintyEvaluator
    cfg["env"]["make_env_function"]["type"]=make_env
    cfg["env"]["evaluator"]["type"] = CartPoleEvaluator
    cfg["env"]["controller"]["type"] =  CartPoleController
    cfg["env"]["observation_manager"]["type"] = CartPoleObservationManager
    cfg["training"]["agent_trainer"]["type"] = CategoricalSACTrainer
    cfg["entry_point"]["type"] = categorical_train_apr
    return cfg
def get_kwargs(dict):
    """
    Gets the kwargs provided in the config files
    Args:
        dict:

    Returns: kwargs

    """
    return {k: v for k, v in dict.items() if k != 'type'}

@hydra.main(version_base=None, config_path=".", config_name="config_cartpole_cheq_with_prior")
def my_app(cfg: DictConfig) -> None:
    cfg = fill_cfg_with_functions(cfg)
    entry_point = cfg["entry_point"]["type"]
    entry_point_kwargs = get_kwargs(cfg["entry_point"])
    entry_point(config=cfg, **entry_point_kwargs)
if __name__ == "__main__":
    my_app()
