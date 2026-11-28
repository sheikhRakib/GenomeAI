import argparse
import os
from datetime import datetime

import pandas as pd

from vep.agents.pipeline import run_agent_pipeline
from vep.baseline import run_scoring, save_results
from vep.llm import DEFAULT_LLM_MODEL, DEFAULT_LLM_URL, build_llm
from vep.model import Evo2Adapter
from vep.validation import run_validation

# each mode is also its flag's attribute name, in run order
MODES = ('baseline', 'single_agent', 'multi_agent')


def parse_args():
    """Parse and check the command line; args.modes lists the chosen modes in run order."""
    ap = argparse.ArgumentParser(description='Run the variant-effect-prediction pipeline.')
    ap.add_argument('path', nargs='?', default='', help='dataset CSV')
    ap.add_argument('--out', default='results', help='parent folder for this run\'s results')
    ap.add_argument('--baseline', action='store_true', help='score directly with Evo2 (non-agent)')
    ap.add_argument('--single-agent', action='store_true',
                    help='single-agent baseline: one LLM-driven agent holds every tool (needs the --llm-url server)')
    ap.add_argument('--multi-agent', action='store_true',
                    help='five LLM-driven agents, each with its own tools (needs the --llm-url server)')
    ap.add_argument('--security', choices=['permissive', 'mock'], default='mock',
                    help='security modules H1-H4 for --single-agent and --multi-agent')
    ap.add_argument('--llm-url', default=DEFAULT_LLM_URL,
                    help='OpenAI-compatible server (vLLM) of the LLM that drives --single-agent and --multi-agent')
    ap.add_argument('--llm-model', default=DEFAULT_LLM_MODEL, help='model the --llm-url server serves')
    ap.add_argument('--cache', action='store_true',
                    help='debugging only: reuse Evo2 outputs saved in data/ by earlier runs; without it every score '
                         'runs Evo2 on both sequences')
    args = ap.parse_args()
    args.modes = [mode for mode in MODES if getattr(args, mode)]
    if not args.modes:
        ap.error('no mode given: add one or more of --baseline, --single-agent, --multi-agent')
    if not args.path.strip():
        ap.error('no dataset given: add the dataset CSV path, e.g. python main.py data/dataset.csv --baseline')
    if not os.path.isfile(args.path):
        ap.error(f'dataset not found: {args.path!r} is not a file; add a valid dataset CSV path')
    return args


def main():
    args = parse_args()
    df = pd.read_csv(args.path)
    failed = run_validation(df, args.path)
    # every agent mode is LLM-driven; building the client fails before any scoring if the server is down
    llm = build_llm(args.llm_url, args.llm_model) if set(args.modes) - {'baseline'} else None
    adapter = Evo2Adapter(cache=args.cache)
    if not args.cache:
        adapter.model  # load now: every score needs Evo2, so a load failure (e.g. GPU memory) stops the run here
    stem = os.path.splitext(os.path.basename(args.path))[0]
    out_dir = os.path.join(args.out, f'{stem}_{datetime.now():%Y%m%d-%H%M%S}')
    try:
        for mode in args.modes:
            if mode == 'baseline':
                valid = df.drop(index=list(failed))  # only validated rows reach the model
                scores, metrics = run_scoring(valid, args.path, adapter)
                save_results(out_dir, df, valid, scores, failed, metrics, args.path, adapter)
            else:
                # every row goes in: the agents validate and reject on their own
                run_agent_pipeline(df, args.path, adapter, f'{out_dir}_{mode}_{args.security}',
                                   security=args.security, mode=mode, llm=llm)
    finally:
        adapter.save_cache()  # with --cache: keep model outputs computed so far, even if the run fails


if __name__ == "__main__":
    main()
