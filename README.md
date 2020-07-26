# playground_hydra_mlflow
- hydra + mlflow を用いた実験管理をシミュレーションしてみる。
- iris をいろんなモデル、パラメータで実験してみるということを行なう。

## 参考
ヤムヤムさん: [ハイパラ管理のすすめ -ハイパーパラメータをHydra+MLflowで管理しよう-](https://ymym3412.hatenablog.com/entry/2020/02/09/034644)

## 実行例
1. デフォルトパラメータで実行 
  `python3 exp_iris.py`

2. パラメータで振る
  `python3 exp_iris.py -m model.params.max_depth=1,5,10,100`

3. モデルの変更
  `python3 exp_iris.py model=logistic_regression`

4. mlflow のwebUI
  `mlflow ui`
