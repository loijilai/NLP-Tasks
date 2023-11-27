import matplotlib.pyplot as plt

ppl = [3.8341, 3.762, 3.7337, 3.7346, 3.7621, 3.8158, 3.9431, 3.9752, 4.0959, 4.147]
train_loss = [5.1429, 4.7813, 4.4612, 4.3351, 4.2236, 4.1655, 4.0748, 3.9666, 3.9253, 3.7954]
step = [225, 450, 675, 900, 1125, 1350, 1575, 1800, 2025, 2250]

# plot learning curves
plt.plot(step, train_loss)
plt.plot(step, ppl)
plt.legend(["train loss", "perplexity"])
plt.xlabel("step")
plt.ylabel("perplexity & train loss")
plt.title("Learning curve of public test set")
plt.savefig("learning_curve.png")