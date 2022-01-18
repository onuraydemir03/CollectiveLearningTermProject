import os
import pandas as pd
import argparse
import os.path as osp
import numpy as np

from simpletransformers.t5 import T5Model

parser = argparse.ArgumentParser(description="Self Training Data Creation Arguments")
parser.add_argument("--dataset_path", default="../../Datasets", help = "Datasets Root Path")
parser.add_argument("--checkpoint_dir", default="../../checkpoints" , help = "Checkpoints root directory")
parser.add_argument("--checkpoint", default="../../checkpoints/Model" , help = "Single checkpoint directory")
parser.add_argument("--eval_batch_size", default = 256)
parser.add_argument("--max_seq_length", default=128)
parser.add_argument("--max_length", default=50)
parser.add_argument("--top_k", default=50)
parser.add_argument("--top_p", default=0.95)


args = parser.parse_args()

# Loads the model
def get_model(args):
    trained_model = T5Model("t5", args.checkpoint, args=args)
    return trained_model

# Predict the paraphrase sentence and returns
def predict(args, sentences):
    trained_model = get_model(args)
    prefix = "paraphrase"
    list_of_test_sentences = [f"{prefix}: {sentence}" for sentence in sentences.iloc[:, 0]]
    pred = trained_model.predict(list_of_test_sentences)
    return pred

# Creates self training data and concats it with original 5000
def create_and_save_self_training_data():
    models = ["T5-Small", "T5-Quora Pretrained"]
    dataset_saving_paths = [model.split("-")[1] for model in models]
    for modelno, model in enumerate(models):
        for datasetNo in range(1, 4):
            args.checkpoint = osp.join(args.checkpoint_dir, model ,  str(datasetNo), "5000")
            base_dataset = pd.read_csv(osp.join(args.dataset_path, str(datasetNo), "Splits", "Train_5000.csv"))
            os.makedirs(osp.join(args.dataset_path, str(datasetNo), "Self Training Datasets",dataset_saving_paths[modelno]), exist_ok=True)
            for X in np.array([1, 2, 3, 4, 5]) * 5000:
                dataset = pd.read_csv(osp.join(args.dataset_path, str(datasetNo), "Splits", str(X) + ".csv"))
                predictions = predict(args, dataset)
                dataset.drop("target_text", axis = 1, inplace = True)
                dataset["target_text"] = predictions
                self_training_x5Dataset = pd.concat([base_dataset, dataset]).reset_index(drop=True)
                self_training_x5Dataset.to_csv(osp.join(args.dataset_path, str(datasetNo), "Self Training Datasets",dataset_saving_paths[modelno], "5X" + str(X // 5000) + ".csv"), index = False)