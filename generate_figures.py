import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotml

df = pd.read_csv("evaluation/fire-2-orig-converted.webm-lables.csv")

print(df)
print(df["predicted"])

arr = df["predicted"]
x = [int(label[1]) for label in arr]
x2 = [int(label[1]) for label in df["actual"]]
print(x)
y = range(0, 2)
print(y)

y = range(0, len(x) )
fig, ax = plt.subplots()
ax.plot(y, x, label="Predicted label")
ax.plot(y, x2, label="Actual label")
ax.legend()

ax.set_yticks([0,1])
ax.set_yticklabels(["No fire", "Fire"])
ax.set_xlabel("Frame number (/ 10)")
ax.set_title("Predicted and actual label for every frame in video:\n LSTM network")
plt.savefig("figures/predicted-v-actual-lstm.png")
plt.close(fig)
conf = np.loadtxt("evaluation/overall_lstm_results.out")

plotml.plot_confusion_matrix(conf, ["No fire", "Fire"])
print(conf)