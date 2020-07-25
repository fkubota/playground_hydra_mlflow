from omegaconf import DictConfig
from util.load_datasets import load_datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


class Runner:
    def __init__(self, client, run_id, X_tr, X_te, y_tr, y_te, cfg):
        self.cfg = cfg
        self.X_tr = X_tr
        self.X_te = X_te
        self.y_tr = y_tr
        self.y_te = y_te
        self.client = client
        self.run_id = run_id

        # method
        # mlflow.set_experiment(self.exp_name)

    def run(self):
        model = self.init_model()
        self.train(model)

    def init_model(self):
        name = self.cfg.model_name
        params = self.cfg.params
        if name == 'RandomForestClassifier':
            model = RandomForestClassifier(**params)
        elif name == 'LogisticRegression':
            model = LogisticRegression(**params)

        return model

    def train(self, model):
        model.fit(self.X_tr, self.y_tr)
        X_tr_pred = model.predict(self.X_tr)
        X_te_pred = model.predict(self.X_te)
        score_tr = accuracy_score(X_tr_pred, self.y_tr)
        score_te = accuracy_score(X_te_pred, self.y_te)

        self.client.log_metric(self.run_id, 'f1_score_train', score_tr)
        self.client.log_metric(self.run_id, 'f1_score_test', score_te)


def main():
    exp_name = 'test'
    config = DictConfig({
            'model_name': 'RandomForestClassifier',
            'params': {'max_depth': 20}
            })
    print(config.pretty())
    X_tr, X_te, y_tr, y_te = load_datasets()
    runner = Runner(exp_name, X_tr, X_te, y_tr, y_te, config)
    runner.run()


if __name__ == "__main__":
    main()
