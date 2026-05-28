import matplotlib.pyplot as plt
import seaborn as sns

def plot_histograms(df):

    df.hist(figsize=(12, 10))

    plt.tight_layout()

    plt.show()


def plot_heatmap(df):

    plt.figure(figsize=(10, 8))

    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")

    plt.show()