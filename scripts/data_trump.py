import argparse
import os
import pickle
import random
import re
import string
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.io import savemat
from sklearn.feature_extraction.text import CountVectorizer


DEFAULT_DATA_URL = "https://drive.google.com/uc?export=download&id=1xRKHaP-QwACMydlDnyFPEaFdtskJuBa6"


def clean_tweet(text):
    text = re.sub(r"http\S+", "", str(text)).lower()
    text = " ".join(token for token in text.split() if not token.startswith("@"))
    text = " ".join(re.sub("[^a-zA-Z]+", " ", text).split())
    return text


def year_month_bin(date_value):
    ts = pd.to_datetime(date_value)
    return f"{ts.year}-{ts.month:02d}"


def remove_empty(in_docs, in_timestamps):
    out_docs = []
    out_timestamps = []
    for ii, doc in enumerate(in_docs):
        if doc:
            out_docs.append(doc)
            out_timestamps.append(in_timestamps[ii])
    return out_docs, out_timestamps


def remove_by_threshold(in_docs, in_timestamps, threshold):
    out_docs = []
    out_timestamps = []
    for ii, doc in enumerate(in_docs):
        if len(doc) > threshold:
            out_docs.append(doc)
            out_timestamps.append(in_timestamps[ii])
    return out_docs, out_timestamps


def create_list_words(in_docs):
    return [x for y in in_docs for x in y]


def create_doc_indices(in_docs):
    aux = [[j for _ in range(len(doc))] for j, doc in enumerate(in_docs)]
    return [int(x) for y in aux for x in y]


def create_bow(doc_indices, words, n_docs, vocab_size):
    return sparse.coo_matrix(([1] * len(doc_indices), (doc_indices, words)), shape=(n_docs, vocab_size)).tocsr()


def split_bow(bow_in, n_docs):
    indices = [[w for w in bow_in[doc, :].indices] for doc in range(n_docs)]
    counts = [[c for c in bow_in[doc, :].data] for doc in range(n_docs)]
    return indices, counts


def write_lda_file(filename, timestamps_in, time_list_in, bow_in):
    idx_sort = np.argsort(timestamps_in)

    with open(filename, "w") as f:
        for row in idx_sort:
            x = bow_in.getrow(row)
            n_elems = x.count_nonzero()
            f.write(str(n_elems))
            for ii, dd in zip(x.indices, x.data):
                f.write(" " + str(ii) + ":" + str(dd))
            f.write("\n")

    with open(filename.replace("-mult", "-seq"), "w") as f:
        f.write(str(len(time_list_in)) + "\n")
        for idx_t, _ in enumerate(time_list_in):
            n_elem = len([t for t in timestamps_in if t == idx_t])
            f.write(str(n_elem) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare Trump tweets for DETM")
    parser.add_argument("--data_url", default=DEFAULT_DATA_URL)
    parser.add_argument("--output_dir", default="data/trump")
    parser.add_argument("--min_df", type=int, default=15)
    parser.add_argument("--max_df", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    print("Loading Trump tweet dataset...")
    trump = pd.read_csv(args.data_url)

    trump["text"] = trump["text"].apply(clean_tweet)
    trump = trump.loc[(trump["isRetweet"] == "f") & (trump["text"] != ""), ["text", "date"]].copy()
    trump["time_bin"] = trump["date"].apply(year_month_bin)

    docs = trump["text"].tolist()
    timestamps = trump["time_bin"].tolist()

    cvectorizer = CountVectorizer(min_df=args.min_df, max_df=args.max_df)
    cvz = cvectorizer.fit_transform(docs).sign()

    sum_counts = cvz.sum(axis=0)
    v_size = sum_counts.shape[1]
    sum_counts_np = np.zeros(v_size, dtype=int)
    for v in range(v_size):
        sum_counts_np[v] = sum_counts[0, v]

    id2word = {idx: word for word, idx in cvectorizer.vocabulary_.items()}
    idx_sort = np.argsort(sum_counts_np)
    vocab = [id2word[idx_sort[cc]] for cc in range(v_size)]

    word2id = {w: j for j, w in enumerate(vocab)}
    all_times = sorted(set(timestamps))
    time2id = {t: i for i, t in enumerate(all_times)}

    num_docs = cvz.shape[0]
    tr_size = int(np.floor(0.85 * num_docs))
    ts_size = int(np.floor(0.10 * num_docs))
    va_size = int(num_docs - tr_size - ts_size)

    idx_permute = np.random.permutation(num_docs).astype(int)

    vocab = list(set([w for idx_d in range(tr_size) for w in docs[idx_permute[idx_d]].split() if w in word2id]))
    word2id = {w: j for j, w in enumerate(vocab)}

    docs_tr = [[word2id[w] for w in docs[idx_permute[idx_d]].split() if w in word2id] for idx_d in range(tr_size)]
    timestamps_tr = [time2id[timestamps[idx_permute[idx_d]]] for idx_d in range(tr_size)]
    docs_ts = [[word2id[w] for w in docs[idx_permute[idx_d + tr_size]].split() if w in word2id] for idx_d in range(ts_size)]
    timestamps_ts = [time2id[timestamps[idx_permute[idx_d + tr_size]]] for idx_d in range(ts_size)]
    docs_va = [
        [word2id[w] for w in docs[idx_permute[idx_d + tr_size + ts_size]].split() if w in word2id]
        for idx_d in range(va_size)
    ]
    timestamps_va = [time2id[timestamps[idx_permute[idx_d + tr_size + ts_size]]] for idx_d in range(va_size)]

    docs_tr, timestamps_tr = remove_empty(docs_tr, timestamps_tr)
    docs_ts, timestamps_ts = remove_empty(docs_ts, timestamps_ts)
    docs_va, timestamps_va = remove_empty(docs_va, timestamps_va)

    docs_ts, timestamps_ts = remove_by_threshold(docs_ts, timestamps_ts, 1)

    docs_ts_h1 = [[w for i, w in enumerate(doc) if i <= len(doc) / 2.0 - 1] for doc in docs_ts]
    docs_ts_h2 = [[w for i, w in enumerate(doc) if i > len(doc) / 2.0 - 1] for doc in docs_ts]

    words_tr = create_list_words(docs_tr)
    words_ts = create_list_words(docs_ts)
    words_ts_h1 = create_list_words(docs_ts_h1)
    words_ts_h2 = create_list_words(docs_ts_h2)
    words_va = create_list_words(docs_va)

    doc_indices_tr = create_doc_indices(docs_tr)
    doc_indices_ts = create_doc_indices(docs_ts)
    doc_indices_ts_h1 = create_doc_indices(docs_ts_h1)
    doc_indices_ts_h2 = create_doc_indices(docs_ts_h2)
    doc_indices_va = create_doc_indices(docs_va)

    n_docs_tr = len(docs_tr)
    n_docs_ts = len(docs_ts)
    n_docs_ts_h1 = len(docs_ts_h1)
    n_docs_ts_h2 = len(docs_ts_h2)
    n_docs_va = len(docs_va)

    bow_tr = create_bow(doc_indices_tr, words_tr, n_docs_tr, len(vocab))
    bow_ts = create_bow(doc_indices_ts, words_ts, n_docs_ts, len(vocab))
    bow_ts_h1 = create_bow(doc_indices_ts_h1, words_ts_h1, n_docs_ts_h1, len(vocab))
    bow_ts_h2 = create_bow(doc_indices_ts_h2, words_ts_h2, n_docs_ts_h2, len(vocab))
    bow_va = create_bow(doc_indices_va, words_va, n_docs_va, len(vocab))

    path_save = os.path.join(args.output_dir, f"min_df_{args.min_df}")
    os.makedirs(path_save, exist_ok=True)

    time_list = [t for t, _ in sorted(time2id.items(), key=lambda x: x[1])]

    write_lda_file(os.path.join(path_save, "dtm_tr-mult.dat"), timestamps_tr, time_list, bow_tr)
    write_lda_file(os.path.join(path_save, "dtm_ts-mult.dat"), timestamps_ts, time_list, bow_ts)
    write_lda_file(os.path.join(path_save, "dtm_ts_h1-mult.dat"), timestamps_ts, time_list, bow_ts_h1)
    write_lda_file(os.path.join(path_save, "dtm_ts_h2-mult.dat"), timestamps_ts, time_list, bow_ts_h2)
    write_lda_file(os.path.join(path_save, "dtm_va-mult.dat"), timestamps_va, time_list, bow_va)

    with open(os.path.join(path_save, "vocab.txt"), "w") as f:
        for v in vocab:
            f.write(v + "\n")

    with open(os.path.join(path_save, "timestamps.txt"), "w") as f:
        for t in time_list:
            f.write(t + "\n")

    with open(os.path.join(path_save, "vocab.pkl"), "wb") as f:
        pickle.dump(vocab, f)

    with open(os.path.join(path_save, "timestamps.pkl"), "wb") as f:
        pickle.dump(time_list, f)

    savemat(os.path.join(path_save, "bow_tr_timestamps.mat"), {"timestamps": timestamps_tr}, do_compression=True)
    savemat(os.path.join(path_save, "bow_ts_timestamps.mat"), {"timestamps": timestamps_ts}, do_compression=True)
    savemat(os.path.join(path_save, "bow_va_timestamps.mat"), {"timestamps": timestamps_va}, do_compression=True)

    bow_tr_tokens, bow_tr_counts = split_bow(bow_tr, n_docs_tr)
    savemat(os.path.join(path_save, "bow_tr_tokens.mat"), {"tokens": bow_tr_tokens}, do_compression=True)
    savemat(os.path.join(path_save, "bow_tr_counts.mat"), {"counts": bow_tr_counts}, do_compression=True)

    bow_ts_tokens, bow_ts_counts = split_bow(bow_ts, n_docs_ts)
    savemat(os.path.join(path_save, "bow_ts_tokens.mat"), {"tokens": bow_ts_tokens}, do_compression=True)
    savemat(os.path.join(path_save, "bow_ts_counts.mat"), {"counts": bow_ts_counts}, do_compression=True)

    bow_ts_h1_tokens, bow_ts_h1_counts = split_bow(bow_ts_h1, n_docs_ts_h1)
    savemat(os.path.join(path_save, "bow_ts_h1_tokens.mat"), {"tokens": bow_ts_h1_tokens}, do_compression=True)
    savemat(os.path.join(path_save, "bow_ts_h1_counts.mat"), {"counts": bow_ts_h1_counts}, do_compression=True)

    bow_ts_h2_tokens, bow_ts_h2_counts = split_bow(bow_ts_h2, n_docs_ts_h2)
    savemat(os.path.join(path_save, "bow_ts_h2_tokens.mat"), {"tokens": bow_ts_h2_tokens}, do_compression=True)
    savemat(os.path.join(path_save, "bow_ts_h2_counts.mat"), {"counts": bow_ts_h2_counts}, do_compression=True)

    bow_va_tokens, bow_va_counts = split_bow(bow_va, n_docs_va)
    savemat(os.path.join(path_save, "bow_va_tokens.mat"), {"tokens": bow_va_tokens}, do_compression=True)
    savemat(os.path.join(path_save, "bow_va_counts.mat"), {"counts": bow_va_counts}, do_compression=True)


    expected_files = [
        "vocab.pkl",
        "timestamps.pkl",
        "bow_tr_tokens.mat",
        "bow_tr_counts.mat",
        "bow_tr_timestamps.mat",
        "bow_va_tokens.mat",
        "bow_va_counts.mat",
        "bow_va_timestamps.mat",
        "bow_ts_tokens.mat",
        "bow_ts_counts.mat",
        "bow_ts_timestamps.mat",
        "bow_ts_h1_tokens.mat",
        "bow_ts_h1_counts.mat",
        "bow_ts_h2_tokens.mat",
        "bow_ts_h2_counts.mat",
    ]
    missing = [f for f in expected_files if not os.path.exists(os.path.join(path_save, f))]
    if missing:
        raise RuntimeError("Preprocessing finished but files are missing: {}".format(missing))

    print(f"Saved DETM-ready Trump dataset to: {path_save}")
    print(f"Documents: train={n_docs_tr}, valid={n_docs_va}, test={n_docs_ts}")
    print(f"Vocabulary size: {len(vocab)}")
    print(f"Time bins: {len(time_list)} ({time_list[0]} .. {time_list[-1]})")


if __name__ == "__main__":
    main()
