"""
Skip-gram with Negative Sampling (SGNS) in pure NumPy.

Two embedding matrices:
  W  (V, D) — center-word embeddings  (kept after training)
  W2 (V, D) — context-word embeddings (discarded after training)

Loss per (center c, positive context o, K negatives n_1..n_K):
  L = -log σ(v_o · u_c) - Σ_k log σ(-v_nk · u_c)

Gradients:
  ∂L/∂u_c   = (σ(pos) - 1)·v_o  + Σ_k σ(neg_k)·v_nk
  ∂L/∂v_o   = (σ(pos) - 1)·u_c
  ∂L/∂v_nk  = σ(neg_k)·u_c
"""

import re
from collections import Counter

import numpy as np


class Vocabulary:
    def __init__(self, min_count: int = 5):
        self.min_count = min_count
        self.word2idx: dict[str, int] = {}
        self.idx2word: list[str] = []
        self.neg_probs: np.ndarray | None = None
        self.keep_probs: np.ndarray | None = None

    def build(self, tokens: list[str], subsample_t: float = 1e-3) -> "Vocabulary":
        counter = Counter(tokens)
        pairs = sorted(
            [(w, c) for w, c in counter.items() if c >= self.min_count],
            key=lambda x: -x[1],
        )
        self.idx2word = [w for w, _ in pairs]
        self.word2idx = {w: i for i, w in enumerate(self.idx2word)}

        total = float(sum(c for _, c in pairs))
        f = np.array([c for _, c in pairs], dtype=np.float64) / total

        # P(w) ∝ freq(w)^0.75
        p75 = f**0.75
        self.neg_probs = (p75 / p75.sum()).astype(np.float64)

        # P(keep) = min(1, (√(f/t) + 1)·(t/f))
        self.keep_probs = np.minimum(
            1.0, (np.sqrt(f / subsample_t) + 1.0) * (subsample_t / f)
        ).astype(np.float32)

        return self

    def __len__(self) -> int:
        return len(self.idx2word)

    def __repr__(self) -> str:
        return f"Vocabulary(size={len(self)}, min_count={self.min_count})"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def encode(tokens: list[str], vocab: Vocabulary) -> np.ndarray:
    out: list[int] = []
    for w in tokens:
        idx = vocab.word2idx.get(w)
        if idx is not None and np.random.random() < vocab.keep_probs[idx]:
            out.append(idx)
    return np.asarray(out, dtype=np.int32)


def pair_batches(ids, window, neg_probs, neg_k, batch_size):
    V = len(neg_probs)
    c_buf: list[int] = []
    o_buf: list[int] = []
    N = len(ids)

    def flush():
        B = len(c_buf)
        neg = np.random.choice(V, size=(B, neg_k), p=neg_probs).astype(np.int32)
        return np.asarray(c_buf, dtype=np.int32), np.asarray(o_buf, dtype=np.int32), neg

    for i in range(N):
        hw = int(np.random.randint(1, window + 1))
        for j in range(max(0, i - hw), min(N, i + hw + 1)):
            if j == i:
                continue
            c_buf.append(ids[i])
            o_buf.append(ids[j])
            if len(c_buf) >= batch_size:
                yield flush()
                c_buf.clear()
                o_buf.clear()

    if c_buf:
        yield flush()


class SkipGramNS:
    def __init__(self, vocab_size: int, embed_dim: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        self.V = vocab_size
        self.D = embed_dim
        self.W = (
            rng.random((vocab_size, embed_dim)).astype(np.float32) - 0.5
        ) / embed_dim
        self.W2 = np.zeros((vocab_size, embed_dim), dtype=np.float32)

    @staticmethod
    def _sigmoid(x):
        return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))

    def sgd_step(self, centers, contexts, negatives, lr) -> float:
        D = self.D
        u_c = self.W[centers]
        v_o = self.W2[contexts]
        v_neg = self.W2[negatives]

        pos_dot = (u_c * v_o).sum(axis=1)
        neg_dots = np.einsum("bd,bkd->bk", u_c, v_neg)

        s_pos = self._sigmoid(pos_dot)
        s_neg = self._sigmoid(-neg_dots)

        loss = (-np.log(s_pos + 1e-7).sum() - np.log(s_neg + 1e-7).sum()) / len(centers)

        d_pos = s_pos - 1.0
        d_neg = 1.0 - s_neg

        grad_u = d_pos[:, None] * v_o + np.einsum("bk,bkd->bd", d_neg, v_neg)
        grad_v_o = d_pos[:, None] * u_c
        grad_v_neg = d_neg[:, :, None] * u_c[:, None, :]

        # np.add.at handles repeated indices correctly (unlike +=)
        np.add.at(self.W, centers, -lr * grad_u)
        np.add.at(self.W2, contexts, -lr * grad_v_o)
        np.add.at(self.W2, negatives.ravel(), -lr * grad_v_neg.reshape(-1, D))

        return float(loss)

    def save(self, path: str) -> None:
        np.savez(path, W=self.W, W2=self.W2)
        print(f"Saved -> {path}.npz")

    @classmethod
    def load(cls, path: str) -> "SkipGramNS":
        data = np.load(path)
        m = cls.__new__(cls)
        m.W, m.W2 = data["W"], data["W2"]
        m.V, m.D = m.W.shape
        return m


def gradient_check(
    vocab_size: int = 20, embed_dim: int = 4, neg_k: int = 3, eps: float = 1e-4
) -> None:
    import copy

    np.random.seed(7)
    model = SkipGramNS(vocab_size, embed_dim, seed=7)

    c = np.array([3], dtype=np.int32)
    o = np.array([5], dtype=np.int32)
    neg = np.array([[1, 2, 4]], dtype=np.int32)

    def loss_fn(m):
        u = m.W[c[0]]
        vo = m.W2[o[0]]
        vn = m.W2[neg[0]]
        return (
            -np.log(m._sigmoid(np.array([float(u @ vo)]))[0] + 1e-7)
            - np.log(m._sigmoid(-(u @ vn.T).astype(float)) + 1e-7).sum()
        )

    def num_grad(mat, idx, d):
        m1 = copy.deepcopy(model)
        getattr(m1, mat)[idx, d] += eps
        m2 = copy.deepcopy(model)
        getattr(m2, mat)[idx, d] -= eps
        return (loss_fn(m1) - loss_fn(m2)) / (2 * eps)

    u_c = model.W[c]
    v_o = model.W2[o]
    v_neg = model.W2[neg]
    pos_d = (u_c * v_o).sum(axis=1)
    neg_d = np.einsum("bd,bkd->bk", u_c, v_neg)
    dp = model._sigmoid(pos_d) - 1.0
    dn = 1.0 - model._sigmoid(-neg_d)

    grad_u = dp[:, None] * v_o + np.einsum("bk,bkd->bd", dn, v_neg)
    grad_vo = dp[:, None] * u_c
    grad_vn = dn[:, :, None] * u_c[:, None, :]

    err_W = max(abs(grad_u[0, d] - num_grad("W", c[0], d)) for d in range(embed_dim))
    err_vo = max(abs(grad_vo[0, d] - num_grad("W2", o[0], d)) for d in range(embed_dim))
    err_vn = max(
        abs(grad_vn[0, k, d] - num_grad("W2", neg[0, k], d))
        for k in range(neg_k)
        for d in range(embed_dim)
    )

    print(f"W  (center)  max err: {err_W:.2e}")
    print(f"W2 (pos ctx) max err: {err_vo:.2e}")
    print(f"W2 (neg ctx) max err: {err_vn:.2e}")
    print("PASSED" if max(err_W, err_vo, err_vn) < 1e-3 else "FAILED")


if __name__ == "__main__":
    gradient_check()
