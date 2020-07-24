import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def load_datasets():
    iris = load_iris()
    data = iris.data
    feat_names = iris.feature_names
    y = iris.target
    df = pd.DataFrame(data, columns=feat_names)

    # train test split
    X_tr, X_te, y_tr, y_te = train_test_split(
                                df, y,
                                test_size=0.33,
                                random_state=42)
    return X_tr, X_te, y_tr, y_te


if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_datasets()
    print(X_tr.head())
    print(y_tr)
