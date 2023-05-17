import numpy as np

from time import time
from bert_score import score
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from nltk.translate import bleu_score
from scipy.stats import pearsonr
from sklearn.model_selection import ParameterGrid

from scoring import get_rouge_scores, get_bleu_scores, \
    get_bert_scores, get_bart_scores, get_bleurt_scores, \
    get_diversity_scores, get_length_scores
from common.bart_score import BARTScorer
from common.evaluation import overall_eval


def build_scores(texts, summaries, args):
    if "rouge_1" in args.metrics_to_use.keys() or "rouge_2" in args.metrics_to_use.keys() or "rouge_l" in args.metrics_to_use.keys():
        rouge_scores = get_rouge_scores(summaries, texts, args)
        r1s, r2s, rls = rouge_scores[0], rouge_scores[1], rouge_scores[2]
    all_scores = []
    for metric in args.metrics_to_use.keys():
        print("Getting scores for metric {}".format(metric))

        # overlap with the source
        if metric == "rouge_1":
            scores = r1s
        elif metric == "rouge_2":
            scores = r2s
        elif metric == "rouge_l":
            scores = rls
        elif metric == "bleu":
            scores = get_bleu_scores(summaries, texts, args)

        # semantic similarity with the source
        elif metric == "bert_score":
            scores = get_bert_scores(summaries, texts, args)
            print("BS", scores.shape)
        elif metric == "bart_score":
            scores = get_bart_scores(summaries, texts, args)
            print("BaS", scores.shape)
        elif metric == "bleurt":
            scores = get_bleurt_scores(summaries, texts, args)
            print("bleurt", scores.shape)

        # intrinsic summary quality
        elif metric == "diversity":
            scores = get_diversity_scores(summaries, args)
        elif metric == "length":
            scores = get_length_scores(texts, summaries, args)

        else:
            raise Exception("Please pass in a valid metric name")
        all_scores.append(np.expand_dims(scores, 1))
    all_scores = np.concatenate(all_scores, 1)
    print("all_scores shape", all_scores.shape)

    return all_scores


def get_manual_weights_idx(all_scores, method, args, weights=[]):
    final_idx, all_candidates_scores = [], []
    if weights == []:
        weights = np.array([args.metrics_to_use[k] for k in args.metrics_to_use.keys()])
    for i in tqdm(range(all_scores.shape[0])):
        all_scores_i = all_scores[i, :, :]
        candidates_scores = []
        for j in range(all_scores_i.shape[1]):
            candidates_scores_j = all_scores_i[:, j]
            if method == "mean":
                candidate_score = np.mean(weights * candidates_scores_j) / np.sum(weights)
            elif method == "gmean":
                candidate_score = 1
                for k in range(len(candidates_scores_j)):
                    candidate_score *= candidates_scores_j[k] ** weights[k]
            candidates_scores.append(candidate_score)
        candidates_scores = np.array(candidates_scores)
        all_candidates_scores.append(candidates_scores)
        sort_idx = np.argsort(candidates_scores)[::-1]
        best_idx = sort_idx[:args.num_beams]
        final_idx.append(best_idx)

    return final_idx, all_candidates_scores


def get_best_grid_weights_hierarchical(all_scores, texts, summaries, labels, args):
    names = list(args.metrics_to_use.keys())

    reduced_all_scores, reduced_names, all_best_weights = [], [], []
    reduced_count = 0
    for i in range(len(args.hierarchical_partition)):
        idx_subset = args.hierarchical_partition[i]
        if len(idx_subset) > 1:
            print("\n" + "*"*50)
            print(f"Running a grid search on the subset {idx_subset}")
            all_scores_i = all_scores[:, idx_subset, :]
            names_i = [names[x] for x in idx_subset]
            best_weights_i = get_best_grid_weights(all_scores_i, names_i, texts, summaries, labels, args)
            all_best_weights.append(list(best_weights_i))
            predicted_scores = np.zeros((all_scores_i.shape[0], all_scores_i.shape[2]))
            for j in range(all_scores_i.shape[1]):
                predicted_scores += all_scores_i[:, j, :] * best_weights_i[j]
            reduced_all_scores.append(np.expand_dims(predicted_scores, 1))
            reduced_names.append(f"reduced_{reduced_count}")
            reduced_count += 1
        else:
            all_best_weights.append([1])
            reduced_all_scores.append(all_scores[:, idx_subset, :])
            reduced_names.append(names[i])
    reduced_all_scores = np.concatenate(reduced_all_scores, 1)

    # global grid search
    print("\n" + "*"*50)
    print("Global grid search")
    best_global_weights = get_best_grid_weights(reduced_all_scores, reduced_names, texts, summaries, labels, args)
    print("\nRecap of best weights")
    print(f"Sets of best weights for each set: {all_best_weights}")
    print(f"Set of best global weights: {best_global_weights}")
    expanded_weights = []
    for i in range(len(all_best_weights)):
        for j in range(len(all_best_weights[i])):
            expanded_weights.append(all_best_weights[i][j] * best_global_weights[i])
    clean_weights = [f"{x:.4f}" for x in expanded_weights]
    print(f"Set of expanded best weights: {clean_weights}")

    return expanded_weights


def get_best_grid_weights(all_scores, names, texts, summaries, labels, args):
    ta = time()

    n = int(1 / args.step) + 1
    vals = list(np.linspace(0, 1, n))
    vals = np.round(vals, 3)
    param_grid = {}
    for name in names:
        param_grid[name] = vals
    grid = ParameterGrid(param_grid)

    all_params = []
    for params in grid:
        # check if that's an allowed combination
        # must sum to 1
        s = np.round(sum([params[k] for k in params.keys()]), 3)
        if s != 1:
            continue
        # some features must be non-
        to_keep = True
        for feats_list in args.non_zero_lists:
            valid = True
            for name in feats_list:
                if not (name in params.keys()):
                    valid = False
                    break
            if valid == False:
                break
            s = sum([params[name] for name in feats_list])
            if s == 0:
                to_keep = False
                break
        if to_keep == False:
            continue
        all_params.append(params)
    print(f"# Coefficient combinations: {len(all_params)}")

    all_weights = [[x[k] for k in x.keys()] for x in all_params]
    all_weights = np.array(all_weights)
    p = np.random.permutation(len(all_weights))
    p = p[:args.n_weights_combinations]
    all_weights = all_weights[p]

    all_mean_scores = []
    best_score = -1
    for i in tqdm(range(len(all_weights))):
        weights = all_weights[i]
        final_idx = []
        for t in range(all_scores.shape[0]):
            all_scores_t = all_scores[t, :, :]
            candidate_scores = []
            for tt in range(all_scores_t.shape[1]):
                candidates_scores_tt = weights * all_scores_t[:, tt]
                candidate_score = np.mean(candidates_scores_tt)
                candidate_scores.append(candidate_score)
            candidate_scores = np.array(candidate_scores)
            best_idx = np.argmax(candidate_scores)
            final_idx.append(best_idx)
        new_summaries = [summaries[k][final_idx[k]] for k in range(len(final_idx))]
        scores, _ = overall_eval(texts, new_summaries, labels, args, display=False)
        # print("Weights: {}, iter {} / {}".format(weights, i + 1, len(all_weights)))
        r1, r2, rl = scores[0], scores[1], scores[2]
        mean_r = (np.mean(r1) + np.mean(r2) + np.mean(rl)) / 3
        all_mean_scores.append(mean_r)
        if mean_r == np.max(np.array(all_mean_scores)):
            print("!!!New best score!!")
            idx = np.argmax(np.array(all_mean_scores))
            print(f"Best mean R: {all_mean_scores[idx]:.2f} for weights: {all_weights[idx]} @ iter {idx+1}"),
    all_mean_scores = np.array(all_mean_scores)
    sort_idx = np.argsort(all_mean_scores)[::-1]
    best_idx = sort_idx[:10]
    best_scores = all_mean_scores[best_idx]
    best_weights = all_weights[best_idx]
    for j in range(len(best_idx)):
        clean_weights = ["{:.4f}".format(x) for x in best_weights[j]]
        print(f"Rank {j+1}, Mean R: {best_scores[j]:.2f}, Weights: {clean_weights}")
    tb = time()
    print(f"Total time: {tb-ta:.2f}, avg time / param set: {(tb - ta) / len(all_weights):.2f}")

    return best_weights[0]


