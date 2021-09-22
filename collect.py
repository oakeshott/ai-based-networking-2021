import pandas as pd

df = pd.read_csv("logs/result.txt", sep="\t", header=None, names=("file", "throughput", "loss_rate"))
print(df.groupby("file").mean())

