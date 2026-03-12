# word2vec-numpy

Word2vec in NumPy with skip-gram and negative sampling

## Setup

```bash
pip install -r requirements.txt
```

## Train

```bash
python train.py                  # WikiText-2 (default)
python train.py --demo           # tiny built-in corpus, no download
python train.py --text file.txt  # custom corpus
```

## Evaluate

```bash
python evaluate.py --emb embeddings.npz --vocab embeddings_vocab.npy --interactive
```

Commands: `nn king` · `sim king queen` · `ana king man woman`

## Gradient check

```bash
python word2vec.py
```
