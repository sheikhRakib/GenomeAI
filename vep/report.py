import json
import os
from datetime import datetime, timezone

from vep.evidence import _py
from vep.validation import _get

REPORT_VERSION = '0.1'
BUILD = 'GRCh38'  # dataset has no build column; ClinVar HGVS uses GRCh38 accessions (NC_*.11 etc.)


def build_report(record, score, model_metadata, evidence, dataset=None, trace=None, security_decisions=None):
    """Result report for one variant: only combines its inputs, never fetches or changes the score."""
    return {'report_version': REPORT_VERSION,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'variant': {'chrom': _get(record, 'CHROM'), 'pos': _py(_get(record, 'POS')),
                        'ref': _get(record, 'REF'), 'alt': _get(record, 'ALT'),
                        'build': BUILD, 'id': _py(_get(record, 'ID'))},
            'prediction': {'score_name': 'neg_delta_ll', 'score': float(score),
                           'interpretation': 'higher = more disruptive', 'uncertainty': None},
            'model': dict(model_metadata),
            'evidence': evidence,
            # pandas row index when given a row; dicts have no index
            'provenance': {'dataset': dataset, 'row_id': _py(getattr(record, 'name', None))},
            'agent_trace': list(trace or []),
            'security_decisions': list(security_decisions or [])}


def write_report(report, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
