
"""This script wrote for the multiple parahprase sentence predictions. But this is not used on this project."""

import numpy as np
from simpletransformers.t5 import T5Model
import os
from nltk.translate.bleu_score import sentence_bleu


root_dir = os.getcwd()

args = {
    "overwrite_output_dir": True,
    "max_seq_length": 256,
    "max_length": 50,
    "top_k": 50,
    "top_p": 0.95,
    "num_return_sequences": 5,
    "eval_batch_size" : 24
}

def get_model(args):
    dir = str(args.split_rate)
    trained_model_path = os.path.join(root_dir, "../checkpoints", dir)
    trained_model = T5Model("t5", trained_model_path, args=args)
    return trained_model

def predict(args, sentences):
    trained_model = get_model(args)
    prefix = "paraphrase"
    list_of_test_sentences = [f"{prefix}: {sentence}" for sentence in sentences]
    pred = trained_model.predict(list_of_test_sentences)

    references = [sentence.split() for sentence in sentences]
    returning_paraphrases = []
    for i, ref in enumerate(references):
        candidates = [pr.split() for pr in pred[i]]
        scores = np.array([sentence_bleu([ref], candidate, weights=(1, 0, 0, 0)) for candidate in candidates])
        most_successful_paraphrase_index = 0
        if len(scores[scores < 1]) :
            most_successful_paraphrase_index = np.argmax(scores[scores < 1 ])
        most_successful_paraphrase = pred[i][most_successful_paraphrase_index]
        returning_paraphrases.append(most_successful_paraphrase)
    return returning_paraphrases
