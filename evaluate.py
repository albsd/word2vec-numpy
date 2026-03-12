"""
Nearest-neighbour and word-analogy queries on trained embeddings.

Usage:
    python evaluate.py --emb embeddings.npz --vocab embeddings_vocab.npy --query king
    python evaluate.py --emb embeddings.npz --vocab embeddings_vocab.npy --analogy "king man woman"
    python evaluate.py --emb embeddings.npz --vocab embeddings_vocab.npy --interactive
"""

import argparse

import numpy as np


def load_embeddings(emb_path: str, vocab_path: str):
    data     = np.load(emb_path)
    W        = data["W"].astype(np.float32)
    idx2word = np.load(vocab_path, allow_pickle=True).tolist()
    word2idx = {w: i for i, w in enumerate(idx2word)}
    norms    = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    return (W / norms).astype(np.float32), idx2word, word2idx


def nearest_neighbours(word, W_norm, idx2word, word2idx, k=10):
    idx = word2idx.get(word)
    if idx is None:
        print(f"'{word}' not in vocabulary.")
        return
    sims = W_norm @ W_norm[idx]
    top  = np.argsort(-sims)[1: k + 1]
    print(f"\nNearest neighbours of '{word}':")
    for rank, i in enumerate(top, 1):
        print(f"  {rank:2d}.  {idx2word[i]:<20s}  {sims[i]:.4f}")


def word_analogy(a, b, c, W_norm, idx2word, word2idx, k=5):
    missing = [w for w in (a, b, c) if w not in word2idx]
    if missing:
        print(f"Not in vocabulary: {missing}")
        return
    query = W_norm[word2idx[b]] - W_norm[word2idx[a]] + W_norm[word2idx[c]]
    norm  = np.linalg.norm(query)
    if norm > 1e-9:
        query /= norm
    sims = W_norm @ query
    for w in (a, b, c):
        sims[word2idx[w]] = -2.0
    top = np.argsort(-sims)[:k]
    print(f"\nAnalogy: '{a}' : '{b}' :: '{c}' : ?")
    for rank, i in enumerate(top, 1):
        print(f"  {rank}.  {idx2word[i]:<20s}  {sims[i]:.4f}")


def cosine_similarity(w1, w2, W_norm, word2idx):
    missing = [w for w in (w1, w2) if w not in word2idx]
    if missing:
        print(f"Not in vocabulary: {missing}")
        return
    print(f"\n  cos('{w1}', '{w2}') = {float(W_norm[word2idx[w1]] @ W_norm[word2idx[w2]]):.4f}")


def interactive(W_norm, idx2word, word2idx):
    print("\nCommands:  nn <word>  |  sim <w1> <w2>  |  ana <a> <b> <c>  |  quit\n")
    while True:
        try:
            parts = input("> ").strip().split()
        except (EOFError, KeyboardInterrupt):
            break
        if not parts:
            continue
        cmd = parts[0].lower()
        if cmd in ("quit", "exit", "q"):
            break
        elif cmd == "nn"  and len(parts) == 2:
            nearest_neighbours(parts[1], W_norm, idx2word, word2idx)
        elif cmd == "sim" and len(parts) == 3:
            cosine_similarity(parts[1], parts[2], W_norm, word2idx)
        elif cmd == "ana" and len(parts) == 4:
            word_analogy(parts[1], parts[2], parts[3], W_norm, idx2word, word2idx)
        else:
            print("  Unknown command.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emb",         required=True)
    ap.add_argument("--vocab",       required=True)
    ap.add_argument("--query",       default=None)
    ap.add_argument("--analogy",     default=None, help='"A B C" for A:B::C:?')
    ap.add_argument("--sim",         default=None, help='"A B" for cosine similarity')
    ap.add_argument("--topk",        type=int, default=10)
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()

    W_norm, idx2word, word2idx = load_embeddings(args.emb, args.vocab)
    print(f"Loaded {len(idx2word):,} embeddings  dim={W_norm.shape[1]}")

    if args.query:
        nearest_neighbours(args.query, W_norm, idx2word, word2idx, args.topk)

    if args.analogy:
        words = args.analogy.strip().split()
        if len(words) != 3: ap.error("--analogy expects exactly three words.")
        word_analogy(*words, W_norm=W_norm, idx2word=idx2word, word2idx=word2idx, k=args.topk)

    if args.sim:
        words = args.sim.strip().split()
        if len(words) != 2: ap.error("--sim expects exactly two words.")
        cosine_similarity(*words, W_norm=W_norm, word2idx=word2idx)

    if args.interactive:
        interactive(W_norm, idx2word, word2idx)

    if not any([args.query, args.analogy, args.sim, args.interactive]):
        for word in ("king", "france", "computer", "music"):
            if word in word2idx:
                nearest_neighbours(word, W_norm, idx2word, word2idx, k=5)
        for trip in [("king", "man", "woman"), ("paris", "france", "germany")]:
            if all(w in word2idx for w in trip):
                word_analogy(*trip, W_norm=W_norm, idx2word=idx2word, word2idx=word2idx)


if __name__ == "__main__":
    main()
