import hashlib
import importlib.metadata
import os
import time

import numpy as np
import torch
from evo2 import Evo2
from evo2.scoring import logits_to_logprobs
from tqdm import tqdm

# A100 has no FP8, so evo2_1b_base (needs Transformer Engine FP8) is unreliable there; 7B runs without it.
DEFAULT_MODEL = 'evo2_7b_base'
DEFAULT_LAYER = 'blocks.28.mlp.l3'
SAVE_EVERY = 200  # new cache entries between saves, so a killed job keeps most of its work


class Evo2Adapter:
    """Frozen Evo2 variant-effect scorer.
    One forward pass per sequence gives both its log-likelihood (sum over positions) and the
    mean-pooled embedding of `layer`. With cache=False (the default, for every reported run) each
    score runs Evo2 on both sequences, every time. cache=True (debugging) reuses outputs saved by
    sequence hash in `cache_dir` and loads the model only when a sequence is not cached yet.
    last_passes describes the two sequences behind the latest score_pair (hash, length, cached or
    not, log-likelihood, time), forward_passes counts the Evo2 forward passes run so far.
    Another model can replace it by offering the same `metadata`, `score_pairs` and `embed_pairs`."""

    def __init__(self, model_name=DEFAULT_MODEL, layer=DEFAULT_LAYER, cache_dir='data', cache=False):
        self.model_name, self.layer, self.cache = model_name, layer, cache
        self.metadata = {'model': model_name, 'evo2_version': importlib.metadata.version('evo2'),
                         'embedding_layer': layer,
                         'score': 'neg_delta_ll = sum LL(wt_seq) - sum LL(mut_seq)', 'cache': cache}
        self.cache_path = os.path.join(cache_dir, f"evo2_cache_{model_name}_{layer.replace('.', '_')}.npz")
        self._model, self._new = None, 0
        self.last_passes, self.forward_passes, self._last = [], 0, {}
        self._cache = {}  # sequence hash -> (log-likelihood, embedding), filled only with cache=True
        if cache and os.path.isfile(self.cache_path):
            z = np.load(self.cache_path)
            self._cache = {k: (float(ll), e) for k, ll, e in zip(z['keys'], z['ll'], z['emb'])}

    @property
    def model(self):
        if self._model is None:
            self._model = Evo2(self.model_name)
        return self._model

    def _forward(self, seq):
        """One forward pass: ((summed log-likelihood, mean-pooled layer embedding), where and how long it ran)."""
        model = self.model  # load outside the timed part
        start = time.perf_counter()
        ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.long, device='cuda:0')[None]
        with torch.inference_mode():
            out, emb = model(ids, return_embeddings=True, layer_names=[self.layer])
            # same log-prob computation as Evo2.score_sequences(reduce_method='sum')
            ll = logits_to_logprobs(out[0], ids).float().cpu().numpy()[0].sum()
        emb = emb[self.layer][0].float().mean(0).cpu().numpy()  # .cpu() waits for the GPU, so the time is real
        self.forward_passes += 1
        ran = {'cached': False, 'device': str(ids.device), 'tokens': ids.shape[1],
               'ms': round(1000 * (time.perf_counter() - start), 1)}
        return (float(ll), emb), ran

    def _get(self, seq):
        seq = seq.strip().upper()
        key = hashlib.sha1(seq.encode()).hexdigest()
        if key in self._cache:
            out, ran = self._cache[key], {'cached': True}
        else:
            out, ran = self._forward(seq)
            if self.cache:
                self._cache[key] = out
                self._new += 1
                if self._new >= SAVE_EVERY:
                    self.save_cache()
        # the sequence itself never leaves: only its hash, so a trace can be checked against the window
        self._last = {'sha1': key, 'bp': len(seq), 'log_likelihood': out[0], **ran}
        return out

    def save_cache(self):
        """Write the cache atomically (temp file + rename), so a crash never leaves a broken file.
        shortcut: two jobs sharing one cache file keep only the last writer's new entries; add file
        locking if runs ever write the same cache in parallel."""
        if not self._new:
            return
        keys = list(self._cache)
        tmp = self.cache_path + '.tmp.npz'
        np.savez(tmp, keys=np.array(keys), ll=np.array([self._cache[k][0] for k in keys]),
                 emb=np.stack([self._cache[k][1] for k in keys]).astype(np.float32))
        os.replace(tmp, self.cache_path)
        self._new = 0

    def log_likelihood(self, seq):
        return self._get(seq)[0]

    def embed(self, seq):
        return self._get(seq)[1]

    def score_pair(self, wt_seq, mut_seq):
        """-Delta log-likelihood; higher means the variant makes the sequence less likely (more disruptive)."""
        wt = self.log_likelihood(wt_seq)
        wt_pass = {'input': 'wt_seq', **self._last}
        mut = self.log_likelihood(mut_seq)
        self.last_passes = [wt_pass, {'input': 'mut_seq', **self._last}]
        return wt - mut

    def score_pairs(self, wt_seqs, mut_seqs):
        scores = [self.score_pair(w, m) for w, m in
                  tqdm(zip(wt_seqs, mut_seqs), total=len(wt_seqs), desc='Scoring', unit='variant')]
        self.save_cache()
        return np.array(scores, dtype=np.float32)

    def embed_pairs(self, wt_seqs, mut_seqs):
        """(wt embeddings, mut embeddings), each [n_variants, hidden_dim]."""
        pairs = [(self.embed(w), self.embed(m)) for w, m in
                 tqdm(zip(wt_seqs, mut_seqs), total=len(wt_seqs), desc='Embedding', unit='variant')]
        self.save_cache()
        return np.stack([p[0] for p in pairs]), np.stack([p[1] for p in pairs])
