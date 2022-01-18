import os

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Visualizes the test bleu scores and saves them as an image
def plot_dataframe(dataframe, model):
    fig = plt.figure(figsize=(7, 7))
    bleu_name = dataframe.columns[-1].capitalize()
    fig.suptitle(bleu_name + " Scores for Model : " + model)
    plt.style.use('ggplot')
    i = 0
    # markers = ["*", "1", "|", "+", "^", "s", "p"]
    # """marker = markers[row]"""
    for row in range(len(dataframe)):
        Y = list(dataframe.iloc[:, -1])
        X = np.arange(len(Y))
        plt.plot(X, Y)
        plt.legend(ncol = 1, loc = "upper right")
        plt.xlabel('Bleu Scores')
        plt.ylabel('Training Sets')
        plt.xticks(X, list(dataframe.iloc[:, 1]  + " " + dataframe.iloc[:, 0]), rotation = 45)
    plt.savefig("../../Visualizations/Test_Results/" + model + "_" + bleu_name + ".png")
    return plt

if __name__ == "__main__":
    test_data = pd.read_csv("../../Visualizations/Test_Results.csv")
    os.makedirs("../../Visualizations/Test_Results/", exist_ok=True)
    for model in test_data["Model"].unique():
        for bleu in range(4):
            df_grouped = test_data[test_data.Model == model].iloc[:,1 : 4 + bleu]
            plot_dataframe(df_grouped.reset_index(drop = True), model)