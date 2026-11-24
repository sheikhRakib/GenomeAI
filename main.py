import argparse
import os
from datetime import datetime

import pandas as pd

from vep.agents.pipeline import run_agent_pipeline
from vep.baseline import run_scoring, save_results
from vep.model import Evo2Adapter
from vep.validation import run_validation


def main():
    ap = argparse.ArgumentParser(description='Run the variant-effect-prediction pipeline.')
    ap.add_argument('path', nargs='?', default='', help='dataset CSV')
    ap.add_argument('--out', default='results', help='parent folder for this run\'s results')
    ap.add_argument('--mode', choices=['baseline', 'agents', 'single'], default='baseline',
                    help='baseline: score directly (non-agent); agents: run the five-agent pipeline; '
                         'single: run the single-agent baseline (one agent with every tool)')
    ap.add_argument('--security', choices=['permissive', 'mock'], default='mock',
                    help='security modules H1-H4 for --mode agents and single')
    args = ap.parse_args()
    if not args.path.strip():
        ap.error('no dataset given: add the dataset CSV path, e.g. python main.py data/dataset.csv')
    if not os.path.isfile(args.path):
        ap.error(f'dataset not found: {args.path!r} is not a file; add a valid dataset CSV path')


    df = pd.read_csv(args.path)
    failed = run_validation(df, args.path)
    adapter = Evo2Adapter()
    stem = os.path.splitext(os.path.basename(args.path))[0]
    out_dir = os.path.join(args.out, f'{stem}_{datetime.now():%Y%m%d-%H%M%S}')
    try:
        if args.mode == 'baseline':
            valid = df.drop(index=list(failed))  # only validated rows reach the model
            scores, metrics = run_scoring(valid, args.path, adapter)
            save_results(out_dir, df, valid, scores, failed, metrics, args.path, adapter)
        else:
            # every row goes in: the agents validate and reject on their own
            run_agent_pipeline(df, args.path, adapter, f'{out_dir}_{args.mode}_{args.security}',
                               security=args.security, mode=args.mode)
    finally:
        adapter.save_cache()  # keep model outputs computed so far, even if the run fails


if __name__ == "__main__":
    main()
