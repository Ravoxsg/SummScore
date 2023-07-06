# quick script to apply SummScore re-ranking on a single data point
import sys
sys.path.append("/data/mathieu/SummScore/src/") # todo: change to your folder path
import torch
import numpy as np
import argparse
from nltk.translate import bleu_score
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoModelForSeq2SeqLM
from bert_score import score

from common.bart_score import BARTScorer
from common.utils import seed_everything


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
args.device = torch.device("cuda")


# seed
seed_everything(42)

# data
dataset_name = "ccdv/cnn_dailymail"
subset = "test"
text_key = "article"
summary_key = "highlights"
dataset = load_dataset(dataset_name, cache_dir="../../hf_datasets/")
dataset = dataset[subset]
texts = dataset[text_key]
p = np.random.permutation(len(texts))
texts = [texts[x] for x in p]
text = texts[0]
text = text.replace("\n", " ")
print("\nSource document:")
print(text)
labels = dataset[summary_key]
labels = [labels[x] for x in p]
label = labels[0]
label = "\n".join(sent_tokenize(label))
print("\nLabel:")
print(label)

# model: we take PEGASUS fine-tuned on XSum, which we transfer to CNN/DM
tokenizer = AutoTokenizer.from_pretrained("google/pegasus-xsum")
model = AutoModelForSeq2SeqLM.from_pretrained("google/pegasus-xsum")

# inference to get summary candidates
tok_text = tokenizer(text, return_tensors="pt", padding="max_length", max_length=1024)
tok_text["input_ids"] = tok_text["input_ids"]
tok_text["attention_mask"] = tok_text["attention_mask"]
with torch.no_grad():
    generated = model.generate(
        input_ids=tok_text["input_ids"].to(args.device),
        attention_mask=tok_text["attention_mask"].to(args.device),
        num_beams=15,
        num_return_sequences=15,
        repetition_penalty=1.0,
        length_penalty=0.8,
        no_repeat_ngram_size=3
    )
candidates = tokenizer.batch_decode(generated, skip_special_tokens=True)
print("\nTop beam:")
print(candidates[0])
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer = True)
rouge_scores = scorer.score(label, candidates[0])
r1 = 100 * rouge_scores["rouge1"].fmeasure
r2 = 100 * rouge_scores["rouge2"].fmeasure
rl = 100 * rouge_scores["rougeLsum"].fmeasure
print(f"Top beam R-1: {r1:.4f}, R-2: {r2:.4f}, R-L: {rl:.4f}")

# build features
all_scores = []
# N-gram overlap features (ROUGE-1, ROUGE-2, BLEU) and intrinsic quality features (diversity, length)
r1s, r2s, bleus, diversities, lengths = [], [], [], [], []
for j in range(len(candidates)):
    # ROUGE features
    rouge_scores = scorer.score(text, candidates[j])
    r1 = 100 * rouge_scores["rouge1"].fmeasure
    r2 = 100 * rouge_scores["rouge2"].fmeasure
    r1s.append(r1)
    r2s.append(r2)
    # BLEU
    bleu_j = bleu_score.sentence_bleu([text.lower().strip().split()], candidates[j].lower().strip().split())
    # BERTScore
    # BARTScore
    # BLEURT
    # Diversity
    words = word_tokenize(candidates[j])
    diverse_1 = len(np.unique(np.array(words)))
    diverse_1 /= max(1, len(words))
    bigrams = [[words[k], words[k + 1]] for k in range(len(words) - 1)]
    diverse_2 = len(np.unique(np.array(bigrams)))
    diverse_2 /= max(1, len(bigrams))
    trigrams = [[words[k], words[k + 1], words[k + 2]] for k in range(len(words) - 2)]
    diverse_3 = len(np.unique(np.array(trigrams)))
    diverse_3 /= max(1, len(trigrams))
    diversity = (diverse_1 + diverse_2 + diverse_3) / 3
    diversities.append(diversity)
    # Length
    length = len(words)
    length = np.abs(args.ratio - length)
    length = 1 / length
    lengths.append(length)
r1s, r2s, bleus, diversities, lengths = np.array(r1s), np.array(r2s), np.array(bleus), np.array(diversities), np.array(lengths)
all_scores.append(np.expand_dims(r1s, 0))
all_scores.append(np.expand_dims(r2s, 0))
all_scores.append(np.expand_dims(bleus, 0))
all_scores.append(np.expand_dims(diversities, 0))
all_scores.append(np.expand_dims(lengths, 0))
# Semantic similarity features (BERTScore, BARTScore, BLEURT)
# BERTScore
p, r, f1 = score(candidates, [text] * len(candidates), lang='en', verbose=False)
bertscores = f1.cpu().numpy()
all_scores.append(np.expand_dims(bertscores, 0))
# BARTScore
bart_scorer = BARTScorer(device = args.device, checkpoint = 'facebook/bart-large-cnn')
bartscores = bart_scorer.score([text] * len(candidates), candidates)
all_scores.append(np.expand_dims(bartscores, 0))
# BLEURT
tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
model = model.cuda()
model.eval()
references = [text] * len(candidates)
inputs = tokenizer(references, candidates, return_tensors="pt", padding=True, truncation=True, max_length=512)
input_ids = inputs["input_ids"].to(args.device)
attention_mask = inputs["attention_mask"].to(args.device)
bleurts = [model(input_ids = input_ids[j:(j+1)], attention_mask = attention_mask[j:(j+1)])[0] for j in range(len(candidates))]
bleurts = np.array(bleurts)
all_scores.append(np.expand_dims(bleurts, 0))
all_scores = np.concatenate(all_scores, 0)
print(all_scores.shape)

# apply re-raking -> can be found in Appendix G of the paper
coeffs = np.array([0.1275, 0.1650, 0.0075, 0.3150, 0.0919, 0.1181, 0.0750, 0.1000])
summscores = all_scores.dot(coeffs)
sort_idx = np.argsort(summscores)[::-1]
top_idx = sort_idx[0]
reranked_candidate = candidates[top_idx]
print(f"\nReranked candidate (idx: {top_idx}):")
print(reranked_candidate)
rouge_scores = scorer.score(label, reranked_candidate)
r1 = 100 * rouge_scores["rouge1"].fmeasure
r2 = 100 * rouge_scores["rouge2"].fmeasure
rl = 100 * rouge_scores["rougeLsum"].fmeasure
print(f"New R-1: {r1:.4f}, R-2: {r2:.4f}, R-L: {rl:.4f}")