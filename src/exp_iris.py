import subprocess
import pathlib
import hydra
from omegaconf import DictConfig
from util.load_datasets import load_datasets
from runner import Runner
import mlflow


EXPERIMENT_NAME = 'iris'


@hydra.main(config_path='../config/config.yaml')
def main(cfg: DictConfig) -> None:
    print(cfg.pretty())

    # hash値の取得
    cmd = "git rev-parse --short HEAD"
    hash_ = subprocess.check_output(cmd.split()).strip().decode('utf-8')

    # 保存先を絶対パスで取得してセット
    cwd = hydra.utils.get_original_cwd()
    path = cwd + '/../mlflow/mlruns'
    path = pathlib.Path(path)
    path = str(path.resolve())
    mlflow.set_tracking_uri(path)
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.log_param('git_hash', hash_)

    # load datasets
    X_tr, X_te, y_tr, y_te = load_datasets()

    # run
    runner = Runner(EXPERIMENT_NAME, X_tr, X_te, y_tr, y_te, cfg)
    runner.run()


if __name__ == "__main__":
    main()
