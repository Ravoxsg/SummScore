import sys
import numpy as np
import pickle

sys.path.append("/data/mathieu/SummScore/src/") # todo: change to your folder path

from nltk.tokenize import sent_tokenize
from tqdm import tqdm
from rouge_score import rouge_scorer

from common.utils import compute_rs



def get_salient_sentences(val_texts, val_labels, args):
    print("Using pseudo labels!! (salient sentences)")
    if args.pseudo_labels == "random":
        val_top_sents = get_random_sentences(val_texts, args)
    elif args.pseudo_labels == "lead":
        val_top_sents = get_lead_sentences(val_texts, args)
    elif args.pseudo_labels == "gsg":
        val_top_sents = get_gsg_sentences(val_texts, args)
    elif args.pseudo_labels == "oracle":
        val_top_sents = get_oracle_sentences(val_texts, val_labels, args)

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer=args.stemmer)
    r1s, r2s, rls = [], [] ,[]
    for i in tqdm(range(len(val_texts))):
        rouge_scores = scorer.score(val_labels[i], val_top_sents[i])
        r1 = 100 * rouge_scores["rouge1"].fmeasure
        r2 = 100 * rouge_scores["rouge2"].fmeasure
        rl = 100 * rouge_scores["rougeLsum"].fmeasure
        r1s.append(r1)
        r2s.append(r2)
        rls.append(rl)
    print(f"\nMean R-1 achieved by the pseudo-labels: {np.mean(r1s):.2f}, R-2: {np.mean(r2s):.2f}, R-L: { np.mean(rls):.2f}")

    return val_top_sents


def get_random_sentences(val_texts, args):
    val_top_sents = []
    for i in tqdm(range(len(val_texts))):
        sents = sent_tokenize(val_texts[i])
        p = np.random.permutation(len(sents))
        idx = p[:args.n_pseudo_sentences]
        idx.sort()
        top_sents = [sents[x] for x in idx]
        top_sents = " ".join(top_sents)
        val_top_sents.append(top_sents)

    return val_top_sents


def get_lead_sentences(val_texts, args):
    val_top_sents = []
    for i in tqdm(range(len(val_texts))):
        sents = sent_tokenize(val_texts[i])
        top_sents = sents[:args.n_pseudo_sentences]
        top_sents = " ".join(top_sents)
        val_top_sents.append(top_sents)

    return val_top_sents


def get_gsg_sentences(val_texts, args):
    path = f"../text_sentences_scores/{args.r_metric}s/{args.dataset}/{args.dataset_name}_{args.val_dataset}_{len(val_texts)}.pkl"
    if args.compute_rs:
        print(f"\nComputing the {args.r_metric}s for {args.val_dataset}")
        val_rs = []
        for i in tqdm(range(len(val_texts))):
            text = val_texts[i]
            if args.truncate_text:
                text = text[:args.max_text_size]
            sents = sent_tokenize(text)
            rs = compute_rs(sents, metric = "rouge{}".format(args.r_metric[1:]))
            val_rs.append(rs)
        with open(path, "wb") as f:
            pickle.dump(val_rs, f)
            print("saved the {}s!".format(args.r_metric))
    else:
        val_rs = pickle.load(open(path, "rb"))
        print(f"loaded the {args.r_metric}s!")
    val_top_sents = []
    for i in tqdm(range(len(val_texts))):
        sents = sent_tokenize(val_texts[i])
        sents = np.array(sents)
        rs = val_rs[i]
        rs = np.array(rs)
        rs = rs[:len(sents)]
        sort_idx = np.argsort(rs)[::-1]
        if args.gsg_ratio >= 1:
            thresh = int(args.gsg_ratio)
        else:
            thresh = int(args.gsg_ratio * len(sort_idx))
        top_idx = sort_idx[:thresh]
        top_idx.sort()
        top_sents = [sents[x] for x in top_idx]
        top_sents = " ".join(top_sents)
        val_top_sents.append(top_sents)

    return val_top_sents


def get_oracle_sentences(val_texts, val_labels, args):
    scorer = rouge_scorer.RougeScorer(['rouge1'], use_stemmer = args.stemmer)
    val_top_sents = []
    for i in tqdm(range(len(val_texts))):
        text_sents = sent_tokenize(val_texts[i])
        if len(text_sents) == 0:
            val_top_sents.append("empty")
            continue
        label_sents = sent_tokenize(val_labels[i])
        used_idx = []
        for j in range(len(label_sents)):
            r1s = []
            for k in range(len(text_sents)):
                if k in used_idx:
                    r1s.append(-1)
                else:
                    rouge_scores = scorer.score(text_sents[k], label_sents[j])
                    r1 = rouge_scores["rouge1"].fmeasure
                    r1s.append(r1)
            r1s = np.array(r1s)
            idx = np.argmax(r1s)
            used_idx.append(idx)
        used_idx.sort()
        top_sents = [text_sents[x] for x in used_idx]
        top_sents = " ".join(top_sents)
        val_top_sents.append(top_sents)

    return val_top_sents
