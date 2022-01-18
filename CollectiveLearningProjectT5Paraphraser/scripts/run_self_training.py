import sklearn.metrics
from simpletransformers.t5 import T5Model

import shutil
import os

import numpy as np
import os.path as osp
import pandas as pd
from sklearn.model_selection import train_test_split
import argparse

parser = argparse.ArgumentParser(description="Self Training Arguments")

parser.add_argument("--dataset_path", default="../../Datasets", help = "Datasets Root Path")
parser.add_argument("--output_dir", default="../../checkpoints" , help = "Saving checkpoints directory")
parser.add_argument("--checkpoint_dir", default="../../checkpoints" , help = "Single checkpoint directory")

args = parser.parse_args()


args_train = {
    "reprocess_input_data": True,
    "overwrite_output_dir": True,
    "max_seq_length": 128,
    "num_train_epochs": 10,
    "num_beams": None,
    "do_sample": True,
    "top_k": 50,
    "top_p": 0.95,
    "use_multiprocessing": False,
    "save_steps": -1,
    "save_eval_checkpoints": True,
    "evaluate_during_training": True,
    "adam_epsilon": 1e-08,
    "eval_batch_size": 14,
    "fp_16": False,
    "gradient_accumulation_steps": 14,
    "learning_rate": 5e-4,
    "max_grad_norm": 1.0,
    "n_gpu": 1,
    "seed": 42,
    "train_batch_size": 14,
    "warmup_steps": 0,
    "weight_decay": 0.0,
    "save_model_every_epoch" : False,
    "use_cuda" : True,
    "output_dir" : "../../checkpoints",
    "early_stopping_metric" : "eval_loss",
    "use_early_stopping" : True
}

models_dict = {
    "Model_Folders" : ["T5-Small", "T5-Quora Pretrained"],
    "Model_Names" : ["t5-small", "mrm8488/t5-small-finetuned-quora-for-paraphrasing"],
    "Dataset_Save_Paths" : ["Small", "Quora Pretrained"]
}
# This function will do the self training operation with created self training datasets
def run_self_training():
    for modelno in range(2): # 2 Models will perform
        for datasetNo in range(1, 4): # Models will perform on 3 different datasets
            for X in np.array([1, 2, 3, 4, 5]) * 5000: # All the datasets have 5 different sized self training datasets
                output_dir = osp.join("../../checkpoints", models_dict["Model_Folders"][modelno], str(datasetNo), "Self Training", "5X" + str(X // 5000))
                os.makedirs(output_dir, exist_ok=True)
                args_train["output_dir"] = output_dir
                model = T5Model("t5", models_dict["Model_Names"][modelno], args=args_train)
                train5kSplit = pd.read_csv(osp.join(args.dataset_path, str(datasetNo), "Self Training Datasets", models_dict["Dataset_Save_Paths"][modelno], "5X" + str(X // 5000) + ".csv"))
                train5kSplit.dropna(axis = 0, inplace = True)
                train5kSplit.columns = ["input_text", "target_text"]
                train5kSplit["prefix"] = "paraphrase"
                train_df, test_df = train_test_split(train5kSplit, test_size=0.2)
                model.train_model(train_df, eval_data=test_df,
                                  use_cuda = True,
                                  acc = sklearn.metrics.accuracy_score,
                                  )
                remove_dirs = [dir for dir in os.listdir(output_dir) if dir.find("checkpoint") != -1]
                for path in remove_dirs:
                    shutil.rmtree(osp.join(output_dir, path), ignore_errors=True)
                os.remove(osp.join(output_dir, "eval_results.txt"))






