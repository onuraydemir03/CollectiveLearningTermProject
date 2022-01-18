import pandas as pd
import argparse
import os.path as osp
from simpletransformers.t5 import T5Model
from nltk.translate.bleu_score import corpus_bleu

parser = argparse.ArgumentParser(description="Test Arguments")

parser.add_argument("--dataset_path", default="../../Datasets", help = "Datasets Root Path")
parser.add_argument("--dataset_name", default="SimpleWiki", help = "Test Dataset Name")
parser.add_argument("--checkpoints_dir", default="../../checkpoints", help = "Root checkpoints directory")
parser.add_argument("--checkpoint", help = "Single model checkpoint directory")
parser.add_argument("--output_dir", default = "../../Visualizations/Test_Results.csv", help = "Save CSV Results Directory")
parser.add_argument("--eval_batch_size", default = 365)
parser.add_argument("--max_seq_length", default=128)
parser.add_argument("--max_length", default=50)
parser.add_argument("--top_k", default=50)
parser.add_argument("--top_p", default=0.95)
args = parser.parse_args()

# Reads the test SimpleWiki dataset
def read_dataset(args):
    full_path = osp.join(args.dataset_path, "Test" ,args.dataset_name + ".csv")
    data = pd.read_csv(full_path)
    return data

# Loads the trained model
def get_model(args):
    trained_model = T5Model("t5", args.checkpoint, args=args)
    return trained_model
# Predicts paraphrase sentence and calculates bleu scores, returns blue1..4
def predict(args, sentences):
    trained_model = get_model(args)
    prefix = "paraphrase"
    list_of_test_sentences = [f"{prefix}: {sentence}" for sentence in sentences.iloc[:, 0]]
    pred = trained_model.predict(list_of_test_sentences)

    references = [sentence.split() for sentence in sentences.iloc[:, 1]]

    bleu1 = corpus_bleu(references, pred, weights = (1.0/1.0, ))
    bleu2 = corpus_bleu(references, pred, weights = (1.0/2.0, 1.0/2.0))
    bleu3 = corpus_bleu(references, pred, weights = (1.0/3.0, 1.0/3.0, 1.0/3.0,))
    bleu4 = corpus_bleu(references, pred)
    return [bleu1, bleu2, bleu3, bleu4]

# Gathers all the results for saving as a comma seperated file
def append_results_to_dict(results, model, splitType, dataset, bleu_scores):
    results["Model"].append(model)
    results["Train Type"].append(splitType)
    results["Dataset"].append(dataset)
    results["bleu-1"].append(bleu_scores[0])
    results["bleu-2"].append(bleu_scores[1])
    results["bleu-3"].append(bleu_scores[2])
    results["bleu-4"].append(bleu_scores[3])
    return results

# Main test function
def test():
    data = read_dataset(args)
    data["prefix"] = "paraphrase"
    data.dropna(axis = 0, inplace = True)
    results = {
        "Model" : [],
        "Train Type" : [],
        "Dataset" : [],
        "bleu-1" : [],
        "bleu-2" : [],
        "bleu-3" : [],
        "bleu-4" : [],
    }
    datasets = {
        1 : "Paws",
        2 : "WikiRow",
        3 : "Tapaco"
    }
    for model in ["T5-Small", "T5-Quora Pretrained"]: # There are 2 models will perform
        for datasetNo in range(1, 4): # There are 3 datasets that models trained on
            for splitType in ["Full", "5000"]: # Full and 5K split are labeled datasets and has 1 model each
                print("Model : ", model, " dataset : ", datasets[datasetNo], "split : ", splitType)
                model_path = osp.join(args.checkpoints_dir, model, str(datasetNo), splitType)
                args.checkpoint = model_path
                bleu_scores = predict(args, data)
                results = append_results_to_dict(results, model, splitType, datasets[datasetNo], bleu_scores)
                pd.DataFrame(results).to_csv(args.output_dir, index=False)
            for X5 in range(1, 6): # 5XK has 5 different self trained models
                splitType = "5X" + str(X5)
                print("Model : ", model, " dataset : ", datasets[datasetNo], "split : ", X5)
                model_path = osp.join(args.checkpoints_dir, model, str(datasetNo), "Self Training", splitType)
                args.checkpoint = model_path
                bleu_scores = predict(args, data)
                results = append_results_to_dict(results, model, splitType, datasets[datasetNo], bleu_scores)
                pd.DataFrame(results).to_csv(args.output_dir, index=False)
    pd.DataFrame(results).to_csv(args.output_dir, index = False)
test()