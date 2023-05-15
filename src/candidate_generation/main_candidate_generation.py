# Generate summary candidates with the fine-tuned models.

import os
import time
import argparse
import sys
import torch
import pickle
import datasets
import openai
from tqdm import tqdm

sys.path.append("/data/mathieu/SummaScore/src/") # todo: change to your folder path

from common.utils import seed_everything
from common.evaluation import overall_eval 
from model_utils import build_tokenizer, build_model
from dataset import Dataset
from engine import beam_search_step



openai.api_key = "xxx" # todo: fill in your OpenAI key here!!

parser = argparse.ArgumentParser()

parser.add_argument('--seed', type = int, default = 42)
parser.add_argument('--cuda', type = bool, default = True)
parser.add_argument('--debug', type = bool, default = False)
parser.add_argument('--debug_size', type = int, default = 10)
parser.add_argument('--few_shot', type = bool, default = True)

# data
parser.add_argument('--dataset', type=str, default = "xsum", choices= ["cnndm", "xsum", "wikihow", "samsum"])

# model
parser.add_argument('--model_type', type = str, default = "pegasus", choices=["pegasus", "bart", "chatgpt"])
parser.add_argument('--model_name', type=str, default = "google/pegasus-xsum",
                    choices = [
                        # Use case #1: Unsupervised abstractive summarization
                        "google/pegasus-large", "gpt-3.5-turbo",

                        # Use case #2: Zero-shot transfer
                        # from CNN/DM
                        "google/pegasus-cnn_dailymail", "facebook/bart-large-cnn", "Yale-LILY/brio-cnndm-cased",
                        # from XSum
                        "google/pegasus-xsum", "facebook/bart-large-xsum", "Yale-LILY/brio-xsum-cased",
                        # from WikiHow
                        "google/pegasus-wikihow", "our_bart_wikihow",
                        # from SAMSum
                        "our_pegasus_samsum", "our_bart_samsum"
                    ])
parser.add_argument('--hidden_size', type = int, default = 768) # 768
parser.add_argument('--cache_dir', type = str, default = "../../../hf_models/pegasus-large/")
parser.add_argument('--load_model_path', type = str, default = "finetuned_checkpoints/our_pegasus_samsum.pt")

# summary generation
parser.add_argument('--val_dataset', type=str, default = "val", choices = ["val", "test"])
parser.add_argument('--max_val_size', type = int, default = 1000)
parser.add_argument('--inference_bs', type = int, default = 2)
parser.add_argument('--save_summaries', type = bool, default = True)
parser.add_argument('--generation_method', type = str, default = "diverse_beam_search",
                    choices = ["beam_search", "diverse_beam_search", "top_p_sampling", "top_k_sampling"])
parser.add_argument('--num_return_sequences', type = int, default = 15) # default: 15
parser.add_argument('--num_beams', type = int, default = 15) # for beam search
parser.add_argument('--num_beam_groups', type = int, default = 15) # for diverse beam search
parser.add_argument('--diversity_penalty', type = float, default = 1.0) # for diverse beam search
parser.add_argument('--top_p', type = float, default = 0.95) # for top-p sampling
parser.add_argument('--top_k', type = int, default = 50) # for top-k sampling
parser.add_argument('--repetition_penalty', type = float, default = 1.0) # for diverse beam search
parser.add_argument('--stemmer', type = bool, default = True)

# metrics
parser.add_argument('--eval_rouge', type = bool, default = True)
parser.add_argument('--eval_bertscore', type = bool, default = False)
parser.add_argument('--eval_bartscore', type = bool, default = False)
parser.add_argument('--eval_new_ngram', type = bool, default = True)
parser.add_argument('--eval_rouge_text', type = bool, default = False)

args = parser.parse_args()


datasets = ["cnndm", "xsum", "wikihow", "samsum"]
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

idx = datasets.index(args.dataset)

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

print("*"*50)
print(args)



def main(args):
    # seed
    seed_everything(args.seed)

    if not(os.path.isdir("../../summaries/")):
        os.makedirs("../../summaries/")
    if not(os.path.isdir("../../summaries/{}/".format(args.dataset))):
        os.makedirs("../../summaries/{}/".format(args.dataset))
    if not(os.path.isdir("../../summaries/{}/{}/".format(args.dataset, args.val_dataset))):
        os.makedirs("../../summaries/{}/{}/".format(args.dataset, args.val_dataset))
    if not(os.path.isdir("../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method))):
        os.makedirs("../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method))

    # device
    device = torch.device("cpu")
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    args.device = device
    print("\nUsing device {}".format(device))

    # data
    dataset_args = [args.dataset_name, args.dataset_version]
    data = datasets.load_dataset(*dataset_args)
    val_data = data["validation"]
    if args.val_dataset == "test":
        val_data = data["test"]
    texts = [val_data[i][args.text_key] for i in range(len(val_data))]
    labels = [val_data[i][args.summary_key] for i in range(len(val_data))]

    # permute
    p = pickle.load(open(f"{args.val_dataset}_permutations/{args.dataset_name}_{args.val_dataset}_permutation.pkl", "rb"))
    texts = [texts[x] for x in p]
    labels = [labels[x] for x in p]

    # sample
    if args.val_dataset == "test" or args.model_type == "chatgpt":
        texts = texts[:args.max_val_size]
        labels = labels[:args.max_val_size]
    if args.debug:
        texts = texts[:args.debug_size]
        labels = labels[:args.debug_size]

    # run the inference
    summaries = []
    # you might have to re-run the script 2-3 times to be able to generate summaries on all datapoints
    if args.model_type == "chatgpt":
        failed_index = []
        for i in tqdm(range(len(texts))):
            text = texts[i]
            prompt = f"Text: {text}.\nSummarize the above text in {args.n} sentence."
            try:
                responsefromgpt = openai.ChatCompletion.create(
                    model=f"{args.gpt_model}",
                    messages=[
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=args.max_summary_length,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    n=args.num_beams
                )
                summaries_i = [responsefromgpt['choices'][j]['message']['content'] for j in range(args.num_beams)]
            except Exception as exc:
                failed_index.append(i)
                summaries_i = []
            summaries.append(summaries_i)
    else:
        # tokenizer
        tokenizer = build_tokenizer(args)
        # datasets
        val_dataset = Dataset(tokenizer, texts, summaries, args)
        print("Total size of dataset: {}".format(len(texts)))
        # data loader
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size = args.inference_bs, shuffle = False)
        # model
        model = build_model(args)
        # loop
        for batch in val_loader:
            summaries_i = beam_search_step(batch, tokenizer, model, args)
            summaries.append(summaries_i)

    # evaluation
    base_results = [summaries[i][0] for i in range(len(summaries))]
    print("*"*100)
    print("\nTop beam:")
    overall_eval(texts, base_results, labels, args)

    # export
    num_candidates = len(val_summaries[0])
    if args.save_summaries:
        path = "../../summaries/{}/{}/{}/".format(args.dataset, args.val_dataset, args.generation_method)
        with open(path + "{}_texts_{}_beams_{}.pkl".format(args.val_dataset, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_texts, f)
        with open(path + "{}_summaries_{}_{}_beams_{}.pkl".format(args.val_dataset, args.model_name, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_summaries, f)
        with open(path + "{}_labels_{}_beams_{}.pkl".format(args.val_dataset, len(val_texts), num_candidates), "wb") as f:
            pickle.dump(val_labels, f)
        print("saved generated summaries!", path)


if __name__ == '__main__':

    main(args)
