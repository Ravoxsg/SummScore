import numpy as np
import pickle
import os

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from bert_score import score
from rouge_score import rouge_scorer
from tqdm import tqdm
from transformers import GPT2Model, GPT2Tokenizer
from scipy.stats import hmean
from scipy.stats.mstats import gmean
from nltk.translate import bleu_score
from nltk.tokenize import word_tokenize

from common.bart_score import BARTScorer
from common.evaluation import overall_eval
from common.summary_processing import pre_rouge_processing


##################################################################### N-gram overlap


def get_rouge_scores(val_summaries, val_pseudo_labels, args):
    print("\nComputing ROUGE scores:")
    if args.compute_rouge:
        os.makedirs(f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/rouge/{args.val_dataset}", exist_ok=True)
        path = f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/rouge/{args.val_dataset}/{args.val_dataset}_rouge_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
        print(f"ROUGE path: {path}")
        all_r1s = np.zeros((len(val_summaries), len(val_summaries[0])))
        all_r2s = np.zeros((len(val_summaries), len(val_summaries[0])))
        all_rls = np.zeros((len(val_summaries), len(val_summaries[0])))
        scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer = args.stemmer)
        for i in tqdm(range(len(val_summaries))):
            val_pseudo_label = val_pseudo_labels[i]
            r1s, r2s, rls = [], [], []
            for j in range(len(val_summaries[i])):
                val_summary = val_summaries[i][j]
                val_summary = pre_rouge_processing(val_summary, args)
                rouge_scores = scorer.score(val_pseudo_label, val_summary)
                r1 = rouge_scores["rouge1"].fmeasure
                r2 = rouge_scores["rouge2"].fmeasure
                rl = rouge_scores["rougeLsum"].fmeasure
                r1s.append(r1)
                r2s.append(r2)
                rls.append(rl)
            r1s = np.array(r1s)
            r2s = np.array(r2s)
            rls = np.array(rls)
            all_r1s[i,:] = r1s
            all_r2s[i,:] = r2s
            all_rls[i,:] = rls
        all_rs = [all_r1s, all_r2s, all_rls]
        with open(path, "wb") as f:
            pickle.dump(all_rs, f)
            print("saved the ROUGE!")
    else:
        all_gen_rs = []
        for x in args.generation_methods:
            path = f"../../summary_scores/{args.dataset_key}/{x}/metrics/rouge/{args.val_dataset}/{args.val_dataset}_rouge_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
            print(f"ROUGE path: {path}")
            with open(path, "rb") as f:
                all_rs = pickle.load(f)
                print("loaded the ROUGE!")
                all_gen_rs.append(all_rs)
        print(len(all_gen_rs))
        print(all_gen_rs[0][0].shape)
        all_r1s = np.concatenate([x[0] for x in all_gen_rs], axis = 1)
        all_r2s = np.concatenate([x[1] for x in all_gen_rs], axis = 1)
        all_rls = np.concatenate([x[2] for x in all_gen_rs], axis = 1)
        print(all_r1s.shape, all_r2s.shape, all_rls.shape)
        all_rs = [all_r1s, all_r2s, all_rls]

    return all_rs


def get_bleu_scores(val_summaries, val_pseudo_labels, args):
    print("\nComputing BLEU scores:")
    if args.compute_bleu:
        os.makedirs(f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/bleu/{args.val_dataset}", exist_ok=True)
        path = f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/bleu/{args.val_dataset}/{args.val_dataset}_bleu_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
        print(f"BLEU path: {path}")
        bleu_scores = []
        for i in tqdm(range(len(val_summaries))):
            val_label = val_pseudo_labels[i]
            bleus_i = []
            for j in range(len(val_summaries[i])):
                val_summary = val_summaries[i][j]
                bleu_i_j = bleu_score.sentence_bleu([val_label.lower().strip().split()], val_summary.lower().strip().split())
                bleus_i.append(bleu_i_j)
            bleus_i = np.array(bleus_i)
            bleu_scores.append(bleus_i)
        bleu_scores = np.array(bleu_scores)
        with open(path, "wb") as f:
            pickle.dump(bleu_scores, f)
            print("saved the BLEU!")
    else:
        all_gen_bleus = []
        for x in args.generation_methods:
            path = f"../../summary_scores/{args.dataset_key}/{x}/metrics/bleu/{args.val_dataset}/{args.val_dataset}_bleu_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
            print(f"BLEU path: {path}")
            with open(path, "rb") as f:
                bleu_scores = pickle.load(f)
                all_gen_bleus.append(bleu_scores)
                print("loaded the ROUGE!")
        bleu_scores = np.concatenate(all_gen_bleus, axis = 1)

    return bleu_scores


##################################################################### Semantic similarity


def get_bert_scores(val_summaries, val_pseudo_labels, args):
    print("\nComputing BERTScore scores:")
    if args.compute_bertscore:
        os.makedirs(
            f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/bertscore/{args.val_dataset}",
            exist_ok=True)
        path = f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/bertscore/{args.val_dataset}/{args.val_dataset}_bertscore_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
        print(f"BS path: {path}")
        bert_scores = []
        top_lengths, selected_lengths = [], []
        for j in range(len(val_summaries[0])):
            val_summaries_j = [val_summaries[i][j] for i in range(len(val_summaries))]
            ### v2
            if args.efficient_bertscore:
                n = args.n_efficient
                block_size = int(len(val_summaries_j)/n)
                all_bs = []
                for k in tqdm(range(n)):
                    block_summaries = val_summaries_j[k*block_size:(k+1)*block_size]
                    block_pseudo_labels = val_pseudo_labels[k*block_size:(k+1)*block_size]
                    p, r, f1 = score(block_summaries, block_pseudo_labels, lang='en', verbose=False)
                    bs = list(f1.cpu().numpy())
                    all_bs += bs
                if n * block_size < len(val_summaries_j):
                    block_summaries = val_summaries_j[n*block_size:]
                    block_pseudo_labels = val_pseudo_labels[n*block_size:]
                    p, r, f1 = score(block_summaries, block_pseudo_labels, lang='en', verbose=False)
                    bs = list(f1.cpu().numpy())
                    all_bs += bs
                with open(f"temp_bs/{args.dataset_name}_{len(val_summaries)}_beams_{args.num_beams}_bertscore_{f}.pkl", "wb") as f:
                    pickle.dump(all_bs, f)
                bert_scores.append(all_bs)
            ### v1
            else:
                p, r, f1 = score(val_summaries_j, val_pseudo_labels, lang='en', verbose=True)
                bert_scores_j = f1.cpu().numpy()
                bert_scores.append(list(bert_scores_j))
        all_bert_scores = []
        for i in range(len(val_summaries)):
            scores_i = [bert_scores[j][i] for j in range(len(bert_scores))]
            #scores_i = [bert_scores[j][i] / len(val_summaries[i][j].split()) for j in range(len(bert_scores))]
            scores_i = np.array(scores_i)
            lengths_i = [len(x.split()) for x in val_summaries[i]]
            top_lengths.append(lengths_i[0])
            selected_lengths.append(lengths_i[np.argmax(scores_i)])
            all_bert_scores.append(scores_i)
        all_bert_scores = np.array(all_bert_scores)
        print(f"Lengths for top beam: {np.mean(top_lengths):.2f}")
        print(f"Lengths for reranked summary: {np.mean(selected_lengths):.2f}")
        with open(path, "wb") as f:
            pickle.dump(all_bert_scores, f)
            print("saved the BS!")
    else:
        all_gen_bs = []
        for x in args.generation_methods:
            path = f"../../summary_scores/{args.dataset_key}/{x}/metrics/bertscore/{args.val_dataset}/{args.val_dataset}_bertscore_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
            print(f"BS path: {path}")
            with open(path, "rb") as f:
                all_bert_scores = pickle.load(f)
                print("loaded the BS!")
                all_gen_bs.append(all_bert_scores)
        all_bert_scores = np.concatenate(all_gen_bs, axis = 1)

    return all_bert_scores


def get_bart_scores(val_summaries, val_pseudo_labels, args):
    print("\nComputing BARTScore scores:")
    if args.compute_bartscore:
        os.makedirs(
            f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/bartscore/{args.val_dataset}",
            exist_ok=True)
        path = f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/bartscore/{args.val_dataset}/{args.val_dataset}_bartscore_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
        print(f"BaS path: {path}")
        bart_scorer = BARTScorer(device = args.device, checkpoint = 'facebook/bart-large-cnn')
        bart_scores = []
        for j in range(len(val_summaries[0])):
            val_summaries_j = [val_summaries[i][j] for i in range(len(val_summaries))]
            bart_scores_j = bart_scorer.score(val_pseudo_labels, val_summaries_j)
            bart_scores.append(bart_scores_j)
        all_bart_scores = []
        for i in range(len(val_summaries)):
            scores_i = [bart_scores[j][i] for j in range(len(bart_scores))]
            scores_i = np.array(scores_i)
            all_bart_scores.append(scores_i)
        all_bart_scores = np.array(all_bart_scores)
        with open(path, "wb") as f:
            pickle.dump(all_bart_scores, f)
            print("saved the BaS!")
    else:
        all_gen_bas = []
        for x in args.generation_methods:
            path = f"../../summary_scores/{args.dataset_key}/{x}/metrics/bartscore/{args.val_dataset}/{args.val_dataset}_bartscore_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
            print(f"BaS path: {path}")
            all_bart_scores = pickle.load(open(path, "rb"))
            print("loaded the BaS!")
            all_gen_bas.append(all_bart_scores)
        all_bart_scores = np.concatenate(all_gen_bas, axis = 1)

    return all_bart_scores


def get_bleurt_scores(val_summaries, val_pseudo_labels, args):
    print("\nComputing BLEURT scores:")
    if args.compute_bleurt:
        os.makedirs(
            f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/bleurt/{args.val_dataset}",
            exist_ok=True)
        path = f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/bleurt/{args.val_dataset}/{args.val_dataset}_bleurt_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
        print(f"BLEURT path: {path}")
        tokenizer = AutoTokenizer.from_pretrained("Elron/bleurt-base-512")
        model = AutoModelForSequenceClassification.from_pretrained("Elron/bleurt-base-512")
        model = model.to(args.device)
        model.eval()
        all_bleurt_scores = []
        for i in tqdm(range(len(val_summaries))):
            summaries_i = val_summaries[i]
            label = val_pseudo_labels[i]
            references = [label] * len(summaries_i)
            inputs = tokenizer(references, summaries_i, return_tensors = "pt", padding = True, truncation = True, max_length = 512)
            input_ids = inputs["input_ids"].to(args.device)
            attention_mask = inputs["attention_mask"].to(args.device)
            scores = []
            for j in range(input_ids.shape[0]):
                scores_j = model(input_ids = input_ids[j:(j+1)], attention_mask = attention_mask[j:(j+1)])[0]
                scores_j = scores_j.detach().cpu().numpy()
                scores.append(scores_j)
            scores = np.concatenate(scores)
            all_bleurt_scores.append(scores[:,0])
        bleurt_scores = np.array(all_bleurt_scores)
        with open(path, "wb") as f:
            pickle.dump(bleurt_scores, f)
            print("saved the BLEURT!")
    else:
        all_gen_brts = []
        for x in args.generation_methods:
            path = f"../../summary_scores/{args.dataset_key}/{x}/metrics/bleurt/{args.val_dataset}/{args.val_dataset}_bleurt_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
            print(f"BLEURT path: {path}")
            bleurt_scores = pickle.load(open(path, "rb"))
            print("loaded the BLEURT!")
            all_gen_brts.append(bleurt_scores)
        bleurt_scores = np.concatenate(all_gen_brts, axis = 1)

    return bleurt_scores


##################################################################### Intrinsic summary quality


def get_diversity_scores(val_summaries, args):
    print("\nComputing diversity scores:")
    if args.compute_diversity:
        os.makedirs(
            f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/diversity/{args.val_dataset}",
            exist_ok=True)
        path = f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/diversity/{args.val_dataset}/{args.val_dataset}_diversity_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
        print(f"Diversity path: {path}")
        diversity_scores = []
        for i in tqdm(range(len(val_summaries))):
            scores = []
            for j in range(len(val_summaries[i])):
                words = word_tokenize(val_summaries[i][j])
                ### v1: unigrams
                diverse_1 = len(np.unique(np.array(words)))
                diverse_1 /= max(1, len(words))
                ### v2: bigrams
                bigrams = [[words[k], words[k+1]] for k in range(len(words)-1)]
                diverse_2 = len(np.unique(np.array(bigrams)))
                diverse_2 /= max(1, len(bigrams))
                ### trigrams
                trigrams = [[words[k], words[k+1], words[k+2]] for k in range(len(words)-2)]
                diverse_3 = len(np.unique(np.array(trigrams)))
                diverse_3 /= max(1, len(trigrams))

                #diverse = diverse_1
                diverse = (diverse_1 + diverse_2 + diverse_3) / 3
                scores.append(diverse)
            scores = np.array(scores)
            scores = np.expand_dims(scores, 0)
            diversity_scores.append(scores)
        diversity_scores = np.concatenate(diversity_scores)
        with open(path, "wb") as f:
            pickle.dump(diversity_scores, f)
            print("saved the diversity!")
    else:
        all_gen_diversity = []
        for x in args.generation_methods:
            path = f"../../summary_scores/{args.dataset_key}/{x}/metrics/diversity/{args.val_dataset}/{args.val_dataset}_diversity_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
            print(f"Diversity path: {path}")
            with open(path, "rb") as f:
                diversity_scores = pickle.load(f)
                print("loaded the diversity!")
                all_gen_diversity.append(diversity_scores)
        diversity_scores = np.concatenate(all_gen_diversity, axis = 1)

    return diversity_scores


def get_length_scores(val_texts, val_summaries, args):
    print("\nComputing length scores:")
    if args.compute_length:
        os.makedirs(
            f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/length/{args.val_dataset}",
            exist_ok=True)
        path = f"../../summary_scores/{args.dataset_key}/{args.generation_methods[0]}/metrics/length/{args.val_dataset}/{args.val_dataset}_length_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
        print(f"Length path: {path}")
        length_scores = []
        for i in tqdm(range(len(val_summaries))):
            text = val_texts[i].lower()
            scores = []
            for j in range(len(val_summaries[i])):
                words = word_tokenize(val_summaries[i][j].lower())
                score_j = len(words)
                score_j = np.abs(args.ratio - score_j)
                score_j = 1 / score_j
                scores.append(score_j)
            scores = np.array(scores)
            scores = np.expand_dims(scores, 0)
            length_scores.append(scores)
        length_scores = np.concatenate(length_scores)
        with open(path, "wb") as f:
            pickle.dump(length_scores, f)
            print("saved the length!")
    else:
        all_gen_lens = []
        for x in args.generation_methods:
            path = f"../../summary_scores/{args.dataset_key}/{x}/metrics/length/{args.val_dataset}/{args.val_dataset}_length_{args.clean_model_name}_{len(val_summaries)}_beams_{args.num_beams}.pkl"
            print(f"Length path: {path}")
            with open(path, "rb") as f:
                length_scores = pickle.load(f)
                print("loaded the length!")
                all_gen_lens.append(length_scores)
        length_scores = np.concatenate(all_gen_lens, axis = 1)

    return length_scores
