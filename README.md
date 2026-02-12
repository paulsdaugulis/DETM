# DETM

This is code that accompanies the paper titled "The Dynamic Embedded Topic Model" by Adji B. Dieng, Francisco J. R. Ruiz, and David M. Blei. (Arxiv link: https://arxiv.org/abs/1907.05545).

The DETM is an extension of the Embedded Topic Model (https://arxiv.org/abs/1907.04907) to corpora with temporal dependencies. The DETM models each word with a categorical distribution whose parameter is given by the inner product between the word embedding and an embedding representation of its assigned topic at a particular time step. The word embeddings allow the DETM to generalize to rare words. The DETM learns smooth topic trajectories by defining a random walk prior over the embeddings of the topics. The DETM is fit using structured amortized variational inference with LSTMs.

## Dependencies

+ python 3.6.7
+ pytorch 1.1.0

## Datasets

The pre-processed UN and ACL datasets can be found below:

+ https://bitbucket.org/franrruiz/data_acl_largev/src/master/
+ https://bitbucket.org/franrruiz/data_undebates_largev/src/master/

The pre-fitted embeddings can be found below:

+ https://bitbucket.org/diengadji/embeddings/src

All the scripts to pre-process a dataset can be found in the folder 'scripts'. 

## Example

To run the DETM on the ACL dataset you can run the command below. You can specify different values for other arguments, peek at the arguments list in main.py.

```
python main.py --dataset acl --data_path PATH_TO_DATA --emb_path PATH_TO_EMBEDDINGS --min_df 10 --num_topics 50 --lr 0.0001 --epochs 1000 --mode train
```


## Citation
```
@article{dieng2019dynamic,
  title={The Dynamic Embedded Topic Model},
  author={Dieng, Adji B and Ruiz, Francisco JR and Blei, David M},
  journal={arXiv preprint arXiv:1907.05545},
  year={2019}
}
```



## Running DETM on Trump's tweets (BERTopic-style example)

Below is a practical workflow to run this DETM implementation on the same Trump tweets dataset used in many BERTopic demos.

### 1) Create a clean Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install numpy scipy pandas scikit-learn matplotlib seaborn torch gensim smart_open
```

> Note: this repository was originally built with older package versions. If you hit compatibility errors, start from Python 3.9/3.10 and install `torch` first.

### 2) Build DETM-format data from Trump tweets

This script downloads the dataset from the Google Drive URL and writes the `.mat` + metadata files expected by `main.py`.

```bash
python scripts/data_trump.py --output_dir data/trump --min_df 15 --max_df 0.7
```

This produces a folder like:

```text
data/trump/min_df_15/
  vocab.pkl
  timestamps.pkl
  bow_tr_tokens.mat
  bow_tr_counts.mat
  bow_tr_timestamps.mat
  ...
```

### 3) Export a common embedding model (GloVe 300d)

The DETM training script expects text embeddings in `word val1 val2 ...` format. Use GloVe (`glove-wiki-gigaword-300`) as a strong/common baseline:

```bash
python scripts/export_glove_embeddings.py \
  --vocab_pkl data/trump/min_df_15/vocab.pkl \
  --output_path data/trump/glove.6B.300d.filtered.txt
```

### 4) Train DETM

```bash
python main.py \
  --dataset trump \
  --data_path data/trump \
  --min_df 15 \
  --emb_path data/trump/glove.6B.300d.filtered.txt \
  --num_topics 30 \
  --emb_size 300 \
  --rho_size 300 \
  --batch_size 256 \
  --lr 0.001 \
  --epochs 200 \
  --mode train
```

### 5) Optional evaluation pass

After training, run with `--mode eval --load_from <checkpoint_path>`.

### Tips for BERTopic users moving to DETM

- BERTopic works on sentence/document embeddings; DETM here works on bag-of-words + **word embeddings**.
- `num_topics` and `min_df` are the first hyperparameters to tune.
- Time granularity is controlled in `scripts/data_trump.py` (currently year-month bins).


### Troubleshooting (Colab)

If training fails with:

```
FileNotFoundError: ... data/trump/min_df_15/bow_tr_tokens.mat
```

it means preprocessing did not finish or wrote to a different location. Run:

```bash
python scripts/data_trump.py --output_dir data/trump --min_df 15 --max_df 0.7
ls -lh data/trump/min_df_15 | head -n 30
```

You should see at least these files:
- `bow_tr_tokens.mat`, `bow_tr_counts.mat`, `bow_tr_timestamps.mat`
- `bow_va_tokens.mat`, `bow_va_counts.mat`, `bow_va_timestamps.mat`
- `bow_ts_tokens.mat`, `bow_ts_counts.mat`, `bow_ts_timestamps.mat`
- `bow_ts_h1_tokens.mat`, `bow_ts_h1_counts.mat`, `bow_ts_h2_tokens.mat`, `bow_ts_h2_counts.mat`
- `vocab.pkl`, `timestamps.pkl`

If you change `--min_df`, make sure the same value is used both in preprocessing and training.
