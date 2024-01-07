import numpy as np
from tqdm import tqdm
from nltk.tokenize import word_tokenize, sent_tokenize
from rouge_score import rouge_scorer
from bert_score import score
from scipy.stats import pearsonr

from common.summary_processing import pre_rouge_processing


def overall_eval(val_texts, val_summaries, val_labels, args, display=True):
    # 1 - ROUGE
    all_score_names = []
    all_scores = []
    if args.eval_rouge:
        r1, r2, rl = rouge_eval("true labels", val_texts, val_summaries, val_labels, args, show_summaries = True, display=display)
        all_scores.append(r1)
        all_scores.append(r2)
        all_scores.append(rl)
        all_score_names += ["ROUGE-1", "ROUGE-2", "ROUGE-L"]
    # 2 - BERTScore
    if args.eval_bertscore:
        bs = bertscore_eval(val_summaries, val_labels, args)
        all_scores.append(bs)
        all_score_names.append("BERTScore")
    # 3 - Abstractiveness
    if args.eval_new_ngram:
        new_ngram_eval(val_texts, val_summaries, args)
    # 6 - Overlap with source
    if args.eval_rouge_text:
        r1_text, r2_text, rl_text = rouge_eval("source", val_summaries, val_texts, val_texts, args)

    return all_scores, all_score_names


def rouge_eval(mode, val_texts, val_summaries, val_labels, args, show_summaries = False, display=True):
    if display:
        print("*"*10, f"1 - ROUGE evaluation with {mode}", "*"*10)
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeLsum'], use_stemmer = args.stemmer)
    all_r1s = []
    all_r2s = []
    all_rls = []
    if display:
        for i in tqdm(range(len(val_summaries))):
            if show_summaries and i < args.n_show_summaries:
                print("*" * 50)
                print(f"\nData point: {i+1} / {len(val_summaries)}")
                print("\nText:")
                print(val_texts[i].replace("\n", " "))
                print("\nLEAD-3:")
                sents = sent_tokenize(val_texts[i])
                lead_3 = " ".join(sents[:3])
                print(lead_3.replace("\n", " "))
                print("\nPredicted summary:")
                print(val_summaries[i].replace("\n", " "))
                print("\nGround-truth summary:")
                print(val_labels[i])

            summary = val_summaries[i]
            summary = pre_rouge_processing(summary, args)
            label = val_labels[i]
            label = pre_rouge_processing(label, args)
            r1, r2, rl = get_rouge_scores(summary, label, scorer, args)
            all_r1s.append(r1)
            all_r2s.append(r2)
            all_rls.append(rl)
    else:
        for i in range(len(val_summaries)):
            if show_summaries and i < args.n_show_summaries:
                print("*" * 50)
                print(f"\nData point: {i+1} / {len(val_summaries)}")
                print("\nText:")
                print(val_texts[i].replace("\n", " "))
                print("\nLEAD-3:")
                sents = sent_tokenize(val_texts[i])
                lead_3 = " ".join(sents[:3])
                print(lead_3.replace("\n", " "))
                print("\nPredicted summary:")
                print(val_summaries[i].replace("\n", " "))
                print("\nGround-truth summary:")
                print(val_labels[i])

            summary = val_summaries[i]
            summary = pre_rouge_processing(summary, args)
            label = val_labels[i]
            label = pre_rouge_processing(label, args)
            r1, r2, rl = get_rouge_scores(summary, label, scorer, args)
            all_r1s.append(r1)
            all_r2s.append(r2)
            all_rls.append(rl)
    all_r1s = 100 * np.array(all_r1s)
    all_r2s = 100 * np.array(all_r2s)
    all_rls = 100 * np.array(all_rls)
    r1 = np.mean(all_r1s)
    r2 = np.mean(all_r2s)
    rl = np.mean(all_rls)
    mean_r = (r1 + r2 + rl) / 3
    if display:
        print(f"Mean R: {mean_r:.4f}, R-1: {r1:.4f} (var: {np.std(all_r1s):.4f}), R-2: {r2:.4f} (var: {np.std(all_r2s):.4f}), R-L: {rl:.4f} (var: {np.std(all_rls):.4f})")

    return all_r1s, all_r2s, all_rls


def get_rouge_scores(summary, label, scorer, args):
    rouge_scores = scorer.score(label, summary)
    r1 = rouge_scores["rouge1"].fmeasure
    r2 = rouge_scores["rouge2"].fmeasure
    rl = rouge_scores["rougeLsum"].fmeasure

    return r1, r2, rl


def bertscore_eval(val_summaries, val_labels, args, verbose=True):
    print("\n", "*" * 10, "2 - BERTScore evaluation", "*" * 10)
    p, r, f1 = score(val_summaries, val_labels, lang='en', verbose=verbose)
    mean_f1 = 100 * f1.mean()
    print("Mean BERTScore F1: {:.2f}".format(mean_f1))
    return 100 * f1.numpy()


def new_ngram_eval(val_texts, val_summaries, args):
    print("\n", "*" * 10, "5 - Abstractiveness / New n-gram", "*" * 10)
    new_unigrams, new_bigrams, new_trigrams, new_quadrigrams = [], [], [], []
    for i in range(len(val_summaries)):
        # text
        text = val_texts[i].lower()
        text_words = word_tokenize(text)
        text_bigrams = [[text_words[j], text_words[j + 1]] for j in range(len(text_words) - 1)]
        text_trigrams = [[text_words[j], text_words[j + 1], text_words[j + 2]] for j in range(len(text_words) - 2)]
        text_quadrigrams = [[text_words[j], text_words[j + 1], text_words[j + 2], text_words[j + 3]] for j in
                            range(len(text_words) - 3)]

        # summary
        summary = val_summaries[i].lower().replace("<n>", " ")
        summary_words = word_tokenize(summary)

        unigrams, bigrams, trigrams, quadrigrams = 0, 0, 0, 0
        for j in range(len(summary_words)):
            if not (summary_words[j] in text_words):
                unigrams += 1
            if j < len(summary_words) - 1:
                bigram = [summary_words[j], summary_words[j + 1]]
                if not (bigram in text_bigrams):
                    bigrams += 1
            if j < len(summary_words) - 2:
                trigram = [summary_words[j], summary_words[j + 1], summary_words[j + 2]]
                if not (trigram in text_trigrams):
                    trigrams += 1
            if j < len(summary_words) - 3:
                quadrigram = [summary_words[j], summary_words[j + 1], summary_words[j + 2], summary_words[j + 3]]
                if not (quadrigram in text_quadrigrams):
                    quadrigrams += 1
        if len(summary_words) > 0:
            new_unigrams.append(unigrams / (len(summary_words) - 0))
        if len(summary_words) > 1:
            new_bigrams.append(bigrams / (len(summary_words) - 1))
        if len(summary_words) > 2:
            new_trigrams.append(trigrams / (len(summary_words) - 2))
        if len(summary_words) > 3:
            new_quadrigrams.append(quadrigrams / (len(summary_words) - 3))
    new_unigrams = np.array(new_unigrams)
    m_uni = 100 * np.mean(new_unigrams)
    new_bigrams = np.array(new_bigrams)
    m_bi = 100 * np.mean(new_bigrams)
    new_trigrams = np.array(new_trigrams)
    m_tri = 100 * np.mean(new_trigrams)
    new_quadrigrams = np.array(new_quadrigrams)
    m_quadri = 100 * np.mean(new_quadrigrams)
    print("New unigrams: {:.2f}, bigrams: {:.2f}, trigrams: {:.2f}, quadrigrams: {:.2f}".format(m_uni, m_bi, m_tri,
                                                                                                m_quadri))


