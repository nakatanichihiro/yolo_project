import pandas as pd

# データフレームを作成
df = pd.DataFrame([
  ["0001", "0.93"],
  ["0002", "0.95"]],
  columns=['time', 'pred'])

df.to_csv("csv_test.csv", index=False, encoding="utf-8", mode='a', header=False)
