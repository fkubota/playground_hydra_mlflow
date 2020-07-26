import pathlib
import hydra
from omegaconf import DictConfig
from util.load_datasets import load_datasets
from run import Runner
import mlflow


# mlflow.set_tracking_uri('./aaa/mlruns')
# EXPERIMENT_NAME = 'iris'
EXPERIMENT_NAME = 'hello'


@hydra.main(config_path='../config/config.yaml')
def main(cfg: DictConfig) -> None:
    cwd = hydra.utils.get_original_cwd()

    # 保存先を絶対パスで取得
    path = cwd + '/../mlflow/mlruns'
    path = pathlib.Path(path)
    path = str(path.resolve())
    mlflow.set_tracking_uri(path)
    mlflow.set_experiment(EXPERIMENT_NAME)
    X_tr, X_te, y_tr, y_te = load_datasets()
    runner = Runner(EXPERIMENT_NAME, X_tr, X_te, y_tr, y_te, cfg)
    runner.run()
    print(cfg.pretty())
    mlflow.log_param('hoge_and_hoge', 'hoge_exp')


if __name__ == "__main__":
    main()
