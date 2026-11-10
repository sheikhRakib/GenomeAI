from vep.validation import _get

SOURCE = 'ClinVar (dataset annotation columns)'
UNTRUSTED_FIELDS = ['conditions', 'clinical_significance', 'hgvs']


def _py(v):
    """Plain Python value: numpy scalars (from pandas rows) unwrapped so json.dumps works."""
    return v.item() if hasattr(v, 'item') else v


def get_evidence(record):
    """Public annotation evidence for one variant record (dict or pandas row).
    Never includes sequences or the labels column."""
    # annotation text is untrusted data: copy it only, never interpret or follow it
    disease = _get(record, 'cln_disease')
    af = _get(record, 'max_af_clinvar')
    return {'source': SOURCE,
            'clinvar_id': _py(_get(record, 'ID')),
            'gene': _get(record, 'gene_name'),
            'clinical_significance': _get(record, 'cln_significance'),
            'conditions': [] if disease is None else
                [c.replace('_', ' ').strip() for c in str(disease).split('|') if c.strip('_ ')],
            'hgvs': _get(record, 'cln_hgvs'),
            'max_allele_frequency': None if af is None else float(af),
            'untrusted_fields': list(UNTRUSTED_FIELDS)}
