"""
Train word2vec (SGNS) in pure NumPy.

Usage:
    python train.py                         # WikiText-2 (default)
    python train.py --embed_dim 200 --epochs 5
    python train.py --text path/to/file.txt
    python train.py --demo                  # tiny built-in corpus, no download
"""

import argparse
import time

import numpy as np

from word2vec import SkipGramNS, Vocabulary, encode, pair_batches, tokenize


def load_wikitext2(split: str = "train") -> list[str]:
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = " ".join(row["text"] for row in ds if row["text"].strip())
    tokens = tokenize(text)
    print(f"Loaded {len(tokens):,} tokens from WikiText-2 ({split}).")
    return tokens


def load_text_file(path: str) -> list[str]:
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    tokens = tokenize(text)
    print(f"Loaded {len(tokens):,} tokens from '{path}'.")
    return tokens


_DEMO_SENTENCES = [
    "the king ruled the kingdom with wisdom and power",
    "the queen ruled the kingdom with grace and beauty",
    "the prince is the son of the king and the queen",
    "the princess is the daughter of the king and the queen",
    "paris is the capital of france and a beautiful city",
    "berlin is the capital of germany and a great city",
    "rome is the capital of italy and an ancient city",
    "madrid is the capital of spain and a vibrant city",
    "man and woman are human beings with equal rights",
    "king man woman queen prince princess royal court",
    "france germany italy spain europe country nation",
    "cat dog animal pet friend loyal companion",
    "book read write learn knowledge school university",
    "water fire earth air nature world planet",
    "music art culture history philosophy science",
] * 200


def load_demo() -> list[str]:
    tokens = tokenize(" ".join(_DEMO_SENTENCES))
    print(f"Demo corpus: {len(tokens):,} tokens.")
    return tokens


def train(
    tokens: list[str],
    embed_dim: int = 100,
    window: int = 5,
    neg_k: int = 5,
    epochs: int = 3,
    batch_size: int = 512,
    lr0: float = 0.025,
    lr_min: float = 1e-4,
    min_count: int = 5,
    subsample_t: float = 1e-3,
    seed: int = 0,
) -> tuple[SkipGramNS, Vocabulary]:
    np.random.seed(seed)

    vocab = Vocabulary(min_count=min_count).build(tokens, subsample_t)
    print(vocab)

    ids = encode(tokens, vocab)
    print(f"  {len(ids):,} tokens after subsampling (from {len(tokens):,}).")

    model = SkipGramNS(len(vocab), embed_dim, seed=seed)
    print(f"\nModel: V={model.V:,}  D={model.D}  params={2 * model.V * model.D:,}\n")

    est_total = len(ids) * window * 2 * epochs
    pairs_done = 0
    t0 = time.time()

    print(
        f"Training  embed_dim={embed_dim}  window={window}  neg_k={neg_k}  "
        f"epochs={epochs}  batch={batch_size}"
    )
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        epoch_pairs = 0

        for centers, contexts, negatives in pair_batches(
            ids, window, vocab.neg_probs, neg_k, batch_size
        ):
            B = len(centers)
            lr = max(lr_min, lr0 * (1.0 - pairs_done / max(est_total, 1)))
            loss = model.sgd_step(centers, contexts, negatives, lr)

            epoch_loss += loss * B
            epoch_pairs += B
            pairs_done += B

            if epoch_pairs % 250_000 < batch_size:
                elapsed = time.time() - t0
                print(
                    f"  epoch {epoch}/{epochs}  pairs {epoch_pairs:>9,}  "
                    f"lr {lr:.5f}  loss {epoch_loss / epoch_pairs:.4f}  "
                    f"{pairs_done / max(elapsed, 1e-9) / 1e3:.0f}k pairs/s  {elapsed:.0f}s"
                )

        print(
            f"-- Epoch {epoch}  pairs {epoch_pairs:,}  "
            f"avg loss {epoch_loss / epoch_pairs:.4f}  "
            f"elapsed {time.time() - t0:.0f}s\n"
        )

    return model, vocab


def main() -> None:
    ap = argparse.ArgumentParser(description="train word2vewc")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text", type=str, help="Path to input file")
    src.add_argument("--demo", action="store_true", help="Tiny built-in corpus.")

    ap.add_argument("--embed_dim", type=int, default=100)
    ap.add_argument("--window", type=int, default=5)
    ap.add_argument("--neg_k", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=0.025)
    ap.add_argument("--min_count", type=int, default=5)
    ap.add_argument("--out", type=str, default="embeddings")
    ap.add_argument("--grad_check", action="store_true")

    args = ap.parse_args()

    if args.grad_check:
        from word2vec import gradient_check

        gradient_check()
        print()

    if args.demo:
        tokens = load_demo()
        args.embed_dim = args.embed_dim if args.embed_dim != 100 else 50
        args.min_count = 1
        args.epochs = args.epochs if args.epochs != 3 else 10
    elif args.text:
        tokens = load_text_file(args.text)
    else:
        tokens = load_wikitext2()

    model, vocab = train(
        tokens,
        embed_dim=args.embed_dim,
        window=args.window,
        neg_k=args.neg_k,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr0=args.lr,
        min_count=args.min_count,
    )

    model.save(args.out)
    np.save(f"{args.out}_vocab.npy", np.array(vocab.idx2word, dtype=object))
    print(f"vocab -> {args.out}_vocab.npy  ({len(vocab):,} words)")


if __name__ == "__main__":
    main()
