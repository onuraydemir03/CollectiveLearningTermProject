import argparse
import os.path as osp
from simpletransformers.t5 import T5Model
from nltk.translate.bleu_score import corpus_bleu

parser = argparse.ArgumentParser(description="Demo Arguments")

parser.add_argument("--checkpoint", help = "Model directory")
parser.add_argument("--sentence", help = "Example sentence")

parser.add_argument("--eval_batch_size", default = 365)
parser.add_argument("--max_seq_length", default=128)
parser.add_argument("--max_length", default=50)
parser.add_argument("--top_k", default=50)
parser.add_argument("--top_p", default=0.95)
parser.add_argument("--num_return_sequences", default=5)

args = parser.parse_args()


def get_model(args):
    trained_model = T5Model("t5", args.checkpoint, args=args)
    return trained_model

def predict(args, sentences):
    trained_model = get_model(args)
    prefix = "paraphrase"
    list_of_test_sentences = [f"{prefix}: {sentence}" for sentence in sentences]
    pred = trained_model.predict(list_of_test_sentences)
    return pred

if __name__ == "__main__":
    args.checkpoint = "/home/onuraydemir/Desktop/Kolektif Proje/checkpoints/T5-Small/3/5000"
    # args.sentence = "I want to be able to swim but there is no pools in my town."
    args.sentence = "I need money, can you lend me some ?"
    preds = predict(args, sentences = [args.sentence])

    for pred in preds[0]:
        print(pred)
