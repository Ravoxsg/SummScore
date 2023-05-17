# Generate summary candidates with the fine-tuned models.

import os
import numpy as np
import argparse
import sys
import torch
import pickle
import datasets
import openai
from tqdm import tqdm

sys.path.append("/data/mathieu/SummScore/src/") # todo: change to your folder path

from common.utils import seed_everything
from engine import build_scores



openai.api_key = "xxx" # todo: fill in your OpenAI key here!!

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = bool, default = True)
parser.add_argument('--debug', type = bool, default = False)
parser.add_argument('--debug_size', type = int, default = 10)
parser.add_argument('--few_shot', type = bool, default = True)

# data
parser.add_argument('--dataset_key', type=str, default = "samsum", choices= ["cnndm", "xsum", "wikihow", "samsum"])
parser.add_argument('--generation_methods', type = list, default = [
    "beam_search",
    #"diverse_beam_search",
    #"top_p_sampling",
    #"top_k_sampling",
])

# model
parser.add_argument('--model_type', type=str, default="pegasus", choices=["pegasus","bart"])
parser.add_argument('--clean_model_name', type=str, default = "pegasus_unsupervised",
                    choices = [
                        # Use case #1: Unsupervised abstractive summarization
                        "pegasus_unsupervised", "chatgpt",

                        # Use case #2: Zero-shot transfer
                        # from CNN/DM
                        "pegasus_cnndm", "bart_cnndm", "brio_cnndm",
                        # from XSum
                        "pegasus_xsum", "bart_xsum", "brio_xsum",
                        # from WikiHow
                        "pegasus_wikihow", "bart_wikihow",
                        # from SAMSum
                        "pegasus_samsum", "bart_samsum"
                    ])

# summary generation
parser.add_argument('--val_dataset', type=str, default = "val", choices = ["val", "test"])
parser.add_argument('--max_val_size', type = int, default = 1000)
parser.add_argument('--num_beams', type = int, default = 20) # for beam search

# features for SummScore
parser.add_argument('--metrics_to_use', type = dict, default = {
    # n-gram overlap with the source
    "rouge_1": 1.0,
    "rouge_2": 1.0,
    "bleu": 1.0,
    # semantic similarity with the source
    "bert_score": 1.0,
    "bart_score": 1.0,
    "bleurt": 1.0,
    # intrinsic summary quality
    "diversity": 1.0,
    "length": 1.0,
})
parser.add_argument('--compute_rouge', type = bool, default = True)
parser.add_argument('--compute_bleu', type = bool, default = True)
parser.add_argument('--compute_bertscore', type = bool, default = True)
parser.add_argument('--efficient_bertscore', type = bool, default = False)
parser.add_argument('--n_efficient', type = int, default = 10)
parser.add_argument('--compute_bartscore', type = bool, default = True)
parser.add_argument('--compute_bleurt', type = bool, default = True)
parser.add_argument('--compute_diversity', type = bool, default = True)
parser.add_argument('--compute_length', type = bool, default = True)
parser.add_argument('--stemmer', type = bool, default = True)

args = parser.parse_args()


dataset_keys = ["cnndm", "xsum", "wikihow", "samsum"]
dataset_names = ["ccdv/cnn_dailymail", "xsum", "wikihow", "samsum"]
dataset_versions = ["3.0.0", "default", "all", "samsum"]
text_keys = ["article", "document", "text", "dialogue"]
summary_keys = ["highlights", "summary", "headline", "summary"]
max_lengths = [1024, 512, 512, 512]
max_summary_lengths = [128, 64, 128, 64]
length_penalties_pegasus = [0.8, 0.8, 0.6, 0.8]
length_penalties_bart = [1.0, 1.0, 1.0, 1.0]
no_repeat_ngram_sizes_pegasus = [0, 3, 0, 0]
no_repeat_ngram_sizes_bart = [3, 3, 3, 3]
ns = [3, 1, 3, 2]
val_sizes = [13368, 11332, 5600, 818]
test_sizes = [11490, 11334, 5580, 819]

idx = dataset_keys.index(args.dataset_key)

args.dataset_name = dataset_names[idx]
args.dataset_version = dataset_versions[idx]
args.text_key = text_keys[idx]
args.summary_key = summary_keys[idx]
args.max_length = max_lengths[idx]
args.max_summary_length = max_summary_lengths[idx]
if args.model_type == "pegasus":
    args.length_penalty = length_penalties_pegasus[idx]
    args.no_repeat_ngram_size = no_repeat_ngram_sizes_pegasus[idx]
elif args.model_type == "bart":
    args.length_penalty = length_penalties_bart[idx]
    args.no_repeat_ngram_size = no_repeat_ngram_sizes_bart[idx]
args.n = ns[idx]
if args.val_dataset == "val":
    args.val_size = val_sizes[idx]
elif args.val_dataset == "test":
    args.val_size = test_sizes[idx]

print("*"*50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    size = min(args.val_size, args.max_val_size)

    # load data
    path = f"../../summaries/{args.dataset_key}/{args.val_dataset}/{args.generation_methods[0]}/"
    texts_path = path + f"{args.val_dataset}_texts_{size}_beams_{args.num_beams}.pkl"
    texts = pickle.load(open(texts_path, "rb"))
    summaries_path = path + f"{args.val_dataset}_summaries_{args.clean_model_name}_{size}_beams_{args.num_beams}.pkl"
    summaries = pickle.load(open(summaries_path, "rb"))

    # build the scores for each summary candidate
    all_scores = build_scores(texts, summaries, args)


if __name__ == '__main__':

    main(args)
