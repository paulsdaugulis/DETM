import argparse
import pickle

import gensim.downloader as api


def main():
    parser = argparse.ArgumentParser(description="Export GloVe vectors for a DETM vocabulary")
    parser.add_argument("--vocab_pkl", required=True)
    parser.add_argument("--output_path", default="data/trump/glove.6B.300d.filtered.txt")
    parser.add_argument("--model_name", default="glove-wiki-gigaword-300")
    args = parser.parse_args()

    with open(args.vocab_pkl, "rb") as f:
        vocab = pickle.load(f)

    print(f"Loading embedding model: {args.model_name}")
    kv = api.load(args.model_name)

    dim = kv.vector_size
    found = 0
    with open(args.output_path, "w", encoding="utf-8") as out:
        for word in vocab:
            if word in kv:
                vec = kv[word]
                out.write(word + " " + " ".join(map(str, vec.tolist())) + "\n")
                found += 1

    print(f"Saved filtered embeddings to: {args.output_path}")
    print(f"Found embeddings for {found}/{len(vocab)} words (dim={dim}).")


if __name__ == "__main__":
    main()
