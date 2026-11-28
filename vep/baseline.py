import json
import os

from sklearn.metrics import average_precision_score, roc_auc_score

# Non-agent baseline pipeline: score validated rows with the model adapter directly, then save results.


def score_metrics(df, scores, name):
    """{'n_scored'} plus zero-shot AUROC/AUPRC when df has both labels; prints them."""
    metrics = {'n_scored': len(scores)}
    if 'labels' in df and df['labels'].nunique() == 2:
        y = df['labels'].astype(int)
        metrics.update(n_positive=int(y.sum()), n_negative=int((1 - y).sum()),
                       auroc=float(roc_auc_score(y, scores)), auprc=float(average_precision_score(y, scores)))
        print(f'{name}: zero-shot AUROC={metrics["auroc"]:.4f} AUPRC={metrics["auprc"]:.4f}')
    return metrics


def run_scoring(df, name, adapter):
    """Score every row with the model adapter; return (scores, metrics), with zero-shot AUROC/AUPRC when labels exist."""
    scores = adapter.score_pairs(df['wt_seq'].tolist(), df['mut_seq'].tolist())
    print(f'{name}: scored {len(scores)} variants with {adapter.metadata}')
    return scores, score_metrics(df, scores, name)


def save_results(out_dir, df, valid, scores, failed, metrics, name, adapter):
    """Write per-variant scores.csv and a metrics.json run summary to a new out_dir."""
    os.makedirs(out_dir)  # new folder per run: never overwrites an earlier run
    cols = [c for c in ('ID', 'CHROM', 'POS', 'REF', 'ALT', 'gene_name', 'labels') if c in valid]
    valid[cols].assign(neg_delta_ll=scores).to_csv(os.path.join(out_dir, 'scores.csv'), index=False)
    summary = {'dataset': name, 'rows': len(df), 'passed': len(valid), 'failed': len(failed),
               'failed_rows': {str(df.at[i, 'ID'] if 'ID' in df else i): [e['code'] for e in errs]
                               for i, errs in failed.items()},
               'model': adapter.metadata, 'cache': adapter.cache_path if adapter.cache else None,
               'evo2_forward_passes': adapter.forward_passes, 'slurm_job_id': os.environ.get('SLURM_JOB_ID'),
               **metrics}
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'{name}: results saved to {out_dir}')
