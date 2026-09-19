from __future__ import annotations

import torch
from evo2 import Evo2
import re
import pandas as pd
import numpy as np


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


def load_data(path: str):
    DNA = re.compile(r'^[ACGTN]+$', re.IGNORECASE)

    df = pd.read_csv(path)

    rows = []
    for _, r in df.iterrows():
        wt, mut = str(r['wt_seq']).upper().strip(), str(r['mut_seq']).upper().strip()
        if wt and mut and DNA.match(wt) and DNA.match(mut) and len(wt) == len(mut):
            rows.append({**r.to_dict(), 'wt_seq': wt, 'mut_seq': mut})
    
    df = pd.DataFrame(rows).reset_index(drop=True)
    labels = df['labels'].astype(int).to_numpy()

    return df, labels


print(get_device())
