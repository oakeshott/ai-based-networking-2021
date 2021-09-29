import pandas as pd

df = pd.read_csv("logs/result.txt", sep="\t", header=None, names=("video_file", "model_path", "throughput", "loss_rate"))
print(df.groupby("video_file").mean()[["throughput", "loss_rate"]])

