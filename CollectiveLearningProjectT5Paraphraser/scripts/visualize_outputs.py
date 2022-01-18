import os
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import os.path as osp

# Plots the training phase train, eval losses per epoch
def plot_dataframe(dataframe, eval_train = "Train"):
    fig = plt.figure()
    dataset_dict = {
        1 : "Paws",
        2 : "WikiRow",
        3 : "Tapaco"
    }
    plot_mode_dict = {
        "Train" : "train_loss",
        "Validation" : "eval_loss"
    }
    fig.suptitle(eval_train + " Loss for Model : " + dataframe.loc[0, "Model"] + " on " + dataset_dict[dataframe.loc[0, "DatasetNo"]] + " Dataset")
    plt.style.use('tableau-colorblind10')
    plotting_list = plot_mode_dict[eval_train]
    for ep, set in enumerate(dataframe["Set"].unique()):
        log_df = dataframe[dataframe.Set == set].iloc[0, -1]
        plt.plot(np.arange(len(log_df)), log_df.loc[:, plotting_list], label = set)
        plt.legend(ncol = 5)
        plt.xlabel('Epochs')
        plt.ylabel('Losses')
        plt.text(0, log_df.loc[0, plot_mode_dict[eval_train]], "{0:.2f}".format(log_df.loc[0, plot_mode_dict[eval_train]]))
        plt.text(len(log_df) -1, log_df.loc[len(log_df) -1, plotting_list], "{0:.2f}".format(log_df.loc[len(log_df) -1, plotting_list]))
        plt.text(5, log_df.loc[5, plotting_list], set)
    return plt

# Saves the loss graphs as an image
def visualize_and_save_training_losses(results_dict, output_dir):
    results_data = pd.DataFrame(results_dict)
    for model in results_data["Model"].unique():
        for datasetNo in results_data["DatasetNo"].unique():
            os.makedirs(osp.join(output_dir, model, str(datasetNo)), exist_ok=True)
            for plot_mode in ["Train", "Validation"]:
                set1 = results_data[np.logical_and(results_data["Model"] == model, results_data["DatasetNo"] == datasetNo)].reset_index(drop = True)
                plt = plot_dataframe(set1, eval_train=plot_mode)
                # plt.show()
                plt.savefig(osp.join(output_dir, model, str(datasetNo), plot_mode + ".png"))


parser = argparse.ArgumentParser(description="Visualize Training Arguments")

parser.add_argument("--checkpoints_dir", default="../../checkpoints", help = "Checkpoints Root Path")
parser.add_argument("--output_dir", default="../../Visualizations", help = "Output directory to save plots.")

if __name__ == "__main__":
    models = ["T5-Small", "T5-Quora Pretrained"]
    results = {
        "Model" : [],
        "DatasetNo" : [],
        "Set": [],
        "LogDF": []
    }
    log_filename = "training_progress_scores.csv"
    args = parser.parse_args()
    for model in models:
        for datasetNo in range(1, 4):
            results["Model"].append(model)
            results["DatasetNo"].append(datasetNo)
            results["Set"].append("5000")
            df = pd.read_csv(osp.join(args.checkpoints_dir, model, str(datasetNo), "5000", log_filename))
            results["LogDF"].append(df)

            results["Model"].append(model)
            results["DatasetNo"].append(datasetNo)
            results["Set"].append("Full")
            df = pd.read_csv(osp.join(args.checkpoints_dir, model, str(datasetNo), "Full", log_filename))
            results["LogDF"].append(df)

            for X in np.array([1, 2, 3, 4, 5]):
                folderName = "5X" + str(X)
                results["Model"].append(model)
                results["DatasetNo"].append(datasetNo)
                results["Set"].append(folderName)
                df = pd.read_csv(osp.join(args.checkpoints_dir, model, str(datasetNo), "Self Training", folderName, log_filename))
                results["LogDF"].append(df)
    print(args.output_dir)
    visualize_and_save_training_losses(results, args.output_dir)