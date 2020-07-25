import hydra
from omegaconf import DictConfig
from util.load_datasets import load_datasets
from run import Runner
import mlflow


# mlflow.set_tracking_uri('./aaa/mlruns')


@hydra.main(config_path='../config/config.yaml')
def main(cfg: DictConfig) -> None:
    client = mlflow.tracking.MlflowClient()
    EXPERIMENT_NAME = 'iris'
    try:
        experiment_id = client.create_experiment(EXPERIMENT_NAME)
    except:
        experiment_id = client.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
    run_id = client.create_run(experiment_id).info.run_id
    print(cfg.pretty())
    X_tr, X_te, y_tr, y_te = load_datasets()
    runner = Runner(client, run_id, X_tr, X_te, y_tr, y_te, cfg)
    # runner = Runner(EXPERIMENT_NAME, X_tr, X_te, y_tr, y_te, cfg)
    runner.run()
    client.log_param(run_id, 'model_name', cfg.model_name)  # LigtGBMのパラメータを記録
    # mlflow.log_param('model_name', cfg.model_name)  # LigtGBMのパラメータを記録


if __name__ == "__main__":
    main()
