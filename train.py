from __future__ import annotations

import torch
import re
import pandas as pd
import numpy as np
from evo2 import Evo2
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score
from pathlib import Path
import os
import torch.nn as nn


os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
# suppress Hugging Face and HTTPX Logs
import logging
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

# Data (each CSV needs wt_seq, mut_seq, labels; start_pos used to center V3 crops)
DATASETS = [
    "data/clinvar_cm_rare_missense_dna_8192.csv",
    # "data/clinvar_cm_missense_dna_8192.csv",
]
MAX_VARIANTS = None            # set e.g. 20 for a quick smoke test

# Shared split
TEST_FRAC, VAL_FRAC, SEED = 0.2, 0.2, 0

# ---- Version 2 (frozen head) ----
V2_MODEL = "evo2_7b_base"          # strong embeddings; cached once
V2_EMB_LAYER = "blocks.28.mlp.l3"  # 7B layer; 1B try "blocks.24.mlp.l3"
V2_FEATURE = "diff"               # 'diff' = mut-wt, 'concat' = [wt,mut], 'mut'
V2_EPOCHS, V2_LR, V2_WD, V2_DROPOUT = 300, 1e-3, 1e-2, 0.3
V2_HIDDEN, V2_BATCH = 128, 128

# ---- Version 3 (LoRA fine-tune) ----
# A100 does NOT support FP8, so evo2_1b_base (needs Transformer Engine FP8) is unreliable.
# Use evo2_7b_base (no TE/FP8 required, same model V2 uses) + LoRA + cropped context.
V3_MODEL = "evo2_7b_base"          # A100-friendly: no FP8/TE needed
V3_EMB_LAYER = "blocks.28.mlp.l3"  # 7B layer (matches V2); printed on failure if wrong
V3_CONTEXT = None                  # crop centered on variant; lower to 1024 if OOM on A100
V3_FEATURE = "diff"                # 'mut' = 1 forward/variant (cheap); 'diff' = 2 forwards
V3_EPOCHS, V3_LR, V3_WD = 3, 1e-4, 1e-2
V3_ACCUM = 8                       # gradient accumulation steps (effective batch)
V3_UNFREEZE_BLOCKS = 2             # fine-tune only the last N Evo2 blocks (raise if memory allows)

# ---- Two-term loss (both versions) ----
# total loss = BCE(embedding head) + SCORE_LOSS_WEIGHT * BCE(calibrated -Delta score);
# final probability = sigmoid(embedding logit + scalar logit)  [Option B: fused logit]
SCORE_LOSS_WEIGHT = 1.0




# get device
def get_device():
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

# set seed for reproducibility
def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)

# regex for DNA sequence
DNA = re.compile(r'^[ACGTN]+$', re.IGNORECASE)


# embed frozen model
def embed_frozen(model, seq, layer):
    device = get_device()
    ids = torch.tensor(model.tokenizer.tokenize(seq), dtype=torch.int)[None].to(device)
    with torch.inference_mode():
        try:
            _, emb = model(ids, return_embeddings=True, layer_names=[layer])
        except Exception as e:
            import torch.nn as nn
            cand = [n for n, m in model.model.named_modules() if 'mlp.l3' in n]
            raise RuntimeError(f"V2_EMB_LAYER '{layer}' failed; try: {cand[:8]}") from e
    return emb[layer][0].float().mean(0).cpu().numpy()



class Head(nn.Module):
    def __init__(self, d, h=V2_HIDDEN, p=V2_DROPOUT):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, h), nn.ReLU(), nn.Dropout(p), nn.Linear(h, 1))
    def forward(self, x): return self.net(x).squeeze(-1)



def fused_prob(Xt, Dt, head, scalar_head):
    return torch.sigmoid(head(Xt) + scalar_head(Dt).squeeze(-1))





def main():
    set_seed(SEED) # set seed for reproducibility

    device = get_device()
    print('Using device:', device)

    # load data
    frames = []
    for p in DATASETS:
        d = pd.read_csv(p); d['__source'] = p; frames.append(d)
    raw = pd.concat(frames, ignore_index=True)
    if MAX_VARIANTS is not None:
        raw = raw.head(MAX_VARIANTS)
    rows = []
    for _, r in raw.iterrows():
        wt, mut = str(r['wt_seq']).upper().strip(), str(r['mut_seq']).upper().strip()
        if wt and mut and DNA.match(wt) and DNA.match(mut) and len(wt) == len(mut):
            rows.append({**r.to_dict(), 'wt_seq': wt, 'mut_seq': mut})
    df = pd.DataFrame(rows).reset_index(drop=True)
    labels = df['labels'].astype(int).to_numpy()

    # split data into train, val, test
    idx = np.arange(len(df))
    tr_idx, te_idx = train_test_split(idx, test_size=TEST_FRAC, stratify=labels, random_state=SEED)
    tr_idx, va_idx = train_test_split(tr_idx, test_size=VAL_FRAC, stratify=labels[tr_idx], random_state=SEED)
    print(f'{len(df)} variants | path={labels.sum()} benign={(labels==0).sum()}')
    print(f'split: train={len(tr_idx)} val={len(va_idx)} test={len(te_idx)}')



    # load cached embeddings if available else generate them
    cache = Path(f"data/evo2_emb_{V2_MODEL}_{V2_EMB_LAYER.replace('.', '_')}.npz")
    if cache.is_file() and len(np.load(cache)['labels']) == len(df):
        z = np.load(cache); wt_emb, mut_emb, neg_delta = z['wt_emb'], z['mut_emb'], z['neg_delta']
        print('Loaded cached embeddings', wt_emb.shape)
    else:
        m2 = Evo2(V2_MODEL)
        wt_l, mut_l, rs_l, as_l, uref = [], [], [], [], {}
        t0 = time.time()
        for i, r in df.iterrows():
            wt, mut = r['wt_seq'], r['mut_seq']
            if wt not in uref:
                uref[wt] = embed_frozen(m2, wt, V2_EMB_LAYER)
            wt_l.append(uref[wt]); mut_l.append(embed_frozen(m2, mut, V2_EMB_LAYER))
            rs_l.append(float(np.array(m2.score_sequences([wt], reduce_method='sum'))[0]))
            as_l.append(float(np.array(m2.score_sequences([mut], reduce_method='sum'))[0]))
            if (i+1) % 25 == 0:
                print(f'  {i+1}/{len(df)} ({(time.time()-t0)/(i+1):.1f}s/variant)')
        wt_emb = np.array(wt_l, np.float32); mut_emb = np.array(mut_l, np.float32)
        neg_delta = (-(np.array(as_l) - np.array(rs_l))).astype(np.float32)
        np.savez_compressed(cache, wt_emb=wt_emb, mut_emb=mut_emb, neg_delta=neg_delta, labels=labels)
        print('Saved', cache, wt_emb.shape)
        del m2; torch.cuda.empty_cache()


    if V2_FEATURE == 'diff':
        feat = mut_emb - wt_emb
    elif V2_FEATURE == 'concat':
        feat = np.hstack([wt_emb, mut_emb])
    else:
        feat = mut_emb
    scaler = StandardScaler().fit(feat[tr_idx])
    def X(sel): return scaler.transform(feat[sel]).astype(np.float32)
    Xtr, Xva, Xte = X(tr_idx), X(va_idx), X(te_idx)
    ytr, yva, yte = labels[tr_idx], labels[va_idx], labels[te_idx]

    # standardized Evo2 pathogenicity score (-Delta) -> second loss term
    delta_scaler = StandardScaler().fit(neg_delta[tr_idx].reshape(-1, 1))
    def D(sel): return delta_scaler.transform(neg_delta[sel].reshape(-1, 1)).astype(np.float32)
    print('feature dim', Xtr.shape[1])





    Xtr_t = torch.tensor(Xtr, device=device); ytr_t = torch.tensor(ytr, dtype=torch.float32, device=device)
    Xva_t = torch.tensor(Xva, device=device); Xte_t = torch.tensor(Xte, device=device)
    Dtr_t = torch.tensor(D(tr_idx), device=device)
    Dva_t = torch.tensor(D(va_idx), device=device); Dte_t = torch.tensor(D(te_idx), device=device)

    head = Head(Xtr.shape[1]).to(device)
    scalar_head = nn.Linear(1, 1).to(device)   # calibrates the Evo2 -Delta score (2nd loss term)
    opt = torch.optim.AdamW(list(head.parameters()) + list(scalar_head.parameters()),
                            lr=V2_LR, weight_decay=V2_WD)
    pw = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(),1)], device=device)
    bce = nn.BCEWithLogitsLoss(pos_weight=pw)


    best_auc, best_state = -1, None
    for ep in range(V2_EPOCHS):
        head.train(); scalar_head.train(); perm = torch.randperm(len(Xtr_t), device=device)
        for b in range(0, len(perm), V2_BATCH):
            s = perm[b:b+V2_BATCH]
            logit_embed = head(Xtr_t[s])
            logit_scalar = scalar_head(Dtr_t[s]).squeeze(-1)
            # two-term loss: diff(embedding) BCE + lambda * pathogenicity-score BCE
            loss = bce(logit_embed, ytr_t[s]) + SCORE_LOSS_WEIGHT * bce(logit_scalar, ytr_t[s])
            opt.zero_grad(); loss.backward(); opt.step()
        head.eval(); scalar_head.eval()
        with torch.no_grad(): pv = fused_prob(Xva_t, Dva_t, head, scalar_head).cpu().numpy()
        auc = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else float('nan')
        if auc > best_auc:
            best_auc = auc
            best_state = ({k: v.clone() for k, v in head.state_dict().items()},
                        {k: v.clone() for k, v in scalar_head.state_dict().items()})
        if (ep+1) % 50 == 0: print(f'epoch {ep+1} loss {loss.item():.4f} val AUROC {auc:.4f}')
    head.load_state_dict(best_state[0]); scalar_head.load_state_dict(best_state[1])
    head.eval(); scalar_head.eval()
    with torch.no_grad(): v2_test = fused_prob(Xte_t, Dte_t, head, scalar_head).cpu().numpy()
    print(f'V2 best val AUROC {best_auc:.4f} | test AUROC {roc_auc_score(yte, v2_test):.4f} '
        f'AUPRC {average_precision_score(yte, v2_test):.4f}')






if __name__ == "__main__":
    main()