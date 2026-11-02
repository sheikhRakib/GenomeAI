from __future__ import annotations

import os
import re
import pandas as pd
import numpy as np
import torch
from evo2 import Evo2
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_curve
from pathlib import Path
import torch.nn as nn
import matplotlib.pyplot as plt



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


def crop_center(seq, center_1based, width):
    if width is None or width >= len(seq): return seq
    half = width // 2
    c = max(0, min(len(seq), center_1based - 1))
    start = max(0, c - half); end = min(len(seq), start + width)
    start = max(0, end - width)
    return seq[start:end]




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



    # Version 3: LoRA fine-tune

    evo3 = Evo2(V3_MODEL)
    base = evo3.model

    # Evo2 loads weights as inference tensors (can't be trained). Clone every parameter/buffer
    # into a normal tensor outside InferenceMode so requires_grad can be set and autograd works.
    _n_conv = 0
    for module in base.modules():
        for pname, p in list(module._parameters.items()):
            if p is not None:
                module._parameters[pname] = nn.Parameter(p.clone(), requires_grad=False); _n_conv += 1
        for bname, b in list(module._buffers.items()):
            if b is not None:
                module._buffers[bname] = b.clone()
    torch.cuda.empty_cache()
    print(f'De-inferenced {_n_conv} parameter tensors (now trainable-capable)')

    for p in base.parameters(): p.requires_grad_(False)
    if hasattr(base, 'gradient_checkpointing_enable'):
        try: base.gradient_checkpointing_enable()
        except Exception: pass

    # unfreeze only the last V3_UNFREEZE_BLOCKS transformer/hyena blocks (by 'blocks.<idx>' name)
    block_idxs = set()
    for n, _ in base.named_parameters():
        m = re.search(r'blocks\.(\d+)\.', n)
        if m: block_idxs.add(int(m.group(1)))
    keep = set(sorted(block_idxs)[-V3_UNFREEZE_BLOCKS:]) if block_idxs else set()
    trainable_names = []
    for n, p in base.named_parameters():
        m = re.search(r'blocks\.(\d+)\.', n)
        if m and int(m.group(1)) in keep:
            p.requires_grad_(True); trainable_names.append(n)
    n_tr = sum(p.numel() for p in base.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in base.parameters())
    print(f'Unfreezing blocks {sorted(keep)} | trainable Evo2 params {n_tr:,} / {n_all:,} '
        f'({100*n_tr/max(n_all,1):.2f}%)')

    # locate embedding layer module for a forward hook
    emb_mod = None
    for n, m in base.named_modules():
        if n.endswith(V3_EMB_LAYER):
            emb_mod = m; break
    if emb_mod is None:
        cand = [n for n, m in base.named_modules() if 'mlp.l3' in n]
        raise RuntimeError(f"V3_EMB_LAYER '{V3_EMB_LAYER}' not found; candidates: {cand[:8]}")
    print('Hooked embedding layer:', V3_EMB_LAYER)

    _cap = {}
    def _hook(mod, inp, out): _cap['h'] = out[0] if isinstance(out, tuple) else out

    _handle = emb_mod.register_forward_hook(_hook)


    def evo_embed_grad(seq):
        ids = torch.tensor(evo3.tokenizer.tokenize(seq), dtype=torch.int)[None].to(device)
        _cap.clear()
        try:
            evo3.model(ids)
        except TypeError:
            evo3.model(input_ids=ids)
        h = _cap['h']                     # [1, L, D]
        return h.float().mean(1).squeeze(0)  # [D]

    def variant_embed(row):
        mut = crop_center(row['mut_seq'], int(row.get('start_pos', len(row['mut_seq'])//2)), V3_CONTEXT)
        if V3_FEATURE == 'diff':
            wt = crop_center(row['wt_seq'], int(row.get('start_pos', len(row['wt_seq'])//2)), V3_CONTEXT)
            return evo_embed_grad(mut) - evo_embed_grad(wt)
        return evo_embed_grad(mut)

    # infer embedding dim with one no-grad pass
    with torch.no_grad():
        _d = variant_embed(df.iloc[int(tr_idx[0])]).shape[0]
    print('V3 embedding dim', _d)




    head3 = nn.Sequential(nn.Linear(_d, 1)).to(device)  # linear head on fine-tuned embeddings
    scalar_head3 = nn.Linear(1, 1).to(device)            # calibrates Evo2 -Delta score (2nd loss term)
    nd_all = delta_scaler.transform(neg_delta.reshape(-1, 1)).astype(np.float32)  # standardized score
    def score_tensor(j):
        return torch.tensor(nd_all[int(j)], device=device)

    params = ([p for p in evo3.model.parameters() if p.requires_grad]
            + list(head3.parameters()) + list(scalar_head3.parameters()))
    opt3 = torch.optim.AdamW(params, lr=V3_LR, weight_decay=V3_WD)
    pw3 = torch.tensor([(ytr == 0).sum()/max((ytr == 1).sum(),1)], device=device)
    bce3 = nn.BCEWithLogitsLoss(pos_weight=pw3)

    def run_split(indices):
        preds = []
        for j in indices:
            e = variant_embed(df.iloc[int(j)])
            fused = head3(e).squeeze(-1) + scalar_head3(score_tensor(j)).squeeze(-1)
            preds.append(torch.sigmoid(fused).item())
        return np.array(preds)

    best_auc3, best3 = -1, None
    for ep in range(V3_EPOCHS):
        evo3.model.train(); head3.train(); scalar_head3.train()
        order = np.random.permutation(tr_idx)
        opt3.zero_grad()
        for step, j in enumerate(order):
            row = df.iloc[int(j)]
            e = variant_embed(row)
            logit_embed = head3(e).squeeze(-1)
            logit_scalar = scalar_head3(score_tensor(j)).squeeze(-1)
            y = torch.tensor([float(labels[int(j)])], device=device)
            # two-term loss: diff(embedding) BCE + lambda * pathogenicity-score BCE
            loss = (bce3(logit_embed.unsqueeze(0), y)
                    + SCORE_LOSS_WEIGHT * bce3(logit_scalar.unsqueeze(0), y)) / V3_ACCUM
            loss.backward()
            if (step+1) % V3_ACCUM == 0:
                opt3.step(); opt3.zero_grad()
        opt3.step(); opt3.zero_grad()
        evo3.model.eval(); head3.eval(); scalar_head3.eval()
        with torch.no_grad(): pv = run_split(va_idx)
        auc = roc_auc_score(yva, pv) if len(np.unique(yva)) > 1 else float('nan')
        if auc > best_auc3:
            best_auc3 = auc
            tr_set = set(trainable_names)
            best3 = ({k: v.detach().clone() for k, v in evo3.model.state_dict().items() if k in tr_set},
                    {k: v.detach().clone() for k, v in head3.state_dict().items()},
                    {k: v.detach().clone() for k, v in scalar_head3.state_dict().items()})
        print(f'V3 epoch {ep+1}/{V3_EPOCHS} val AUROC {auc:.4f}')

    if best3 is not None:
        finetune_sd, head_sd, scalar_sd = best3
        evo3.model.load_state_dict(finetune_sd, strict=False)
        head3.load_state_dict(head_sd); scalar_head3.load_state_dict(scalar_sd)
    evo3.model.eval(); head3.eval(); scalar_head3.eval()
    with torch.no_grad(): v3_test = run_split(te_idx)
    print(f'V3 best val AUROC {best_auc3:.4f} | test AUROC {roc_auc_score(yte, v3_test):.4f} '
        f'AUPRC {average_precision_score(yte, v3_test):.4f}')
    _handle.remove()



    def boot_ci(y, p, metric=roc_auc_score, n=2000):
        rs = np.random.default_rng(SEED); vals = []
        for _ in range(n):
            s = rs.integers(0, len(y), len(y))
            if len(np.unique(y[s])) < 2: continue
            vals.append(metric(y[s], p[s]))
        return np.percentile(vals, [2.5, 97.5])

    zs_test = neg_delta[te_idx]
    rows_tbl = []
    for name, p in [('zero-shot -d', zs_test), ('V2 frozen head', v2_test), ('V3 LoRA finetune', v3_test)]:
        lo, hi = boot_ci(yte, p)
        rows_tbl.append({'model': name, 'AUROC': roc_auc_score(yte, p),
                        'AUROC_lo': lo, 'AUROC_hi': hi, 'AUPRC': average_precision_score(yte, p)})
    print(pd.DataFrame(rows_tbl).to_string(index=False))

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for name, p in [('zero-shot -d', zs_test), ('V2 frozen', v2_test), ('V3 LoRA', v3_test)]:
        fpr, tpr, _ = roc_curve(yte, p)
        ax[0].plot(fpr, tpr, label=f'{name}, AUROC={roc_auc_score(yte, p):.3f}')
    ax[0].plot([0,1],[0,1],'k--',alpha=.4); ax[0].set_xlabel('FPR'); ax[0].set_ylabel('TPR')
    ax[0].set_title('Test ROC'); ax[0].legend(loc='lower right')
    ax[1].hist(v2_test[yte==0], bins=20, alpha=.6, label='benign'); ax[1].hist(v2_test[yte==1], bins=20, alpha=.6, label='pathogenic')
    ax[1].set_title('V2 head output (path->1, benign->0)'); ax[1].set_xlabel('prob'); ax[1].legend()
    plt.tight_layout(); fig.savefig('data/cm_evo2_test_roc_v2_hist.png', dpi=150); plt.close(fig)
    print('Wrote data/cm_evo2_test_roc_v2_hist.png')


    out = df.copy()
    out['neg_delta_likelihood'] = neg_delta
    split = np.array(['train']*len(df), dtype=object); split[va_idx]='val'; split[te_idx]='test'
    out['split'] = split
    p2 = np.full(len(df), np.nan); p3 = np.full(len(df), np.nan)
    with torch.no_grad():
        p2[te_idx] = v2_test; p3[te_idx] = v3_test
        p2[va_idx] = fused_prob(Xva_t, Dva_t, head, scalar_head).cpu().numpy()
        p2[tr_idx] = fused_prob(Xtr_t, Dtr_t, head, scalar_head).cpu().numpy()
    out['v2_prob'] = p2; out['v3_prob'] = p3
    out.to_csv('data/cm_evo2_head_scores_v2_v3.csv', index=False)
    torch.save({'head': head.state_dict(), 'scalar_head': scalar_head.state_dict()}, 'data/v2_head.pt')
    print('Wrote data/cm_evo2_head_scores_v2_v3.csv and data/v2_head.pt')


if __name__ == "__main__":
    main()