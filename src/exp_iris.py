import hydra
from omegaconf import DictConfig
from util.load_datasets import load_datasets
from run import Runner
import mlflow


# mlflow.set_tracking_uri('./aaa/mlruns')
EXPERIMENT_NAME = 'iris'


@hydra.main(config_path='../config/config.yaml')
def main(cfg: DictConfig) -> None:
    cwd = hydra.utils.get_original_cwd()
    mlflow.set_tracking_uri(cwd + '/../mlflow')
    mlflow.log_param('model_name', cfg.model_name)  # LigtGBMのパラメータを記録
    X_tr, X_te, y_tr, y_te = load_datasets()
    runner = Runner(EXPERIMENT_NAME, X_tr, X_te, y_tr, y_te, cfg)


if __name__ == "__main__":
    main()
