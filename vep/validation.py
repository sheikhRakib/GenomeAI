import math
import re
from collections import Counter

from tqdm import tqdm


# Each check takes a record (dict or pandas row) and returns an error dict, or None if it passes.
# A check skips (returns None) when its inputs are missing or malformed: those are reported once
# by check_required_fields / the format checks, so every check is also safe to call alone.


VALID_CHROMS = {str(i) for i in range(1, 23)} | {'X', 'Y', 'M'}
ALLELE = re.compile(r'^[ACGT]+$')
SEQ = re.compile(r'^[ACGTN]+$')
COMPLEMENT = str.maketrans('ACGT', 'TGCA')

PATHOGENIC = {'Pathogenic', 'Likely_pathogenic', 'Pathogenic/Likely_pathogenic'}
BENIGN = {'Benign', 'Likely_benign', 'Benign/Likely_benign'}

# genomic HGVS forms present in the ClinVar data: SNV, delins, inversion
HGVS_SNV = re.compile(r'^NC_0*(\d+)\.\d+:g\.(\d+)([ACGT])>([ACGT])$')
HGVS_DELINS = re.compile(r'^NC_0*(\d+)\.\d+:g\.(\d+)_(\d+)delins([ACGT]+)$')
HGVS_INV = re.compile(r'^NC_0*(\d+)\.\d+:g\.(\d+)_(\d+)inv$')
ACCESSION_CHROM = {'23': 'X', '24': 'Y', '12920': 'M'}  # RefSeq accession number -> chrom

MAX_WINDOW = 8192  # longest sequence window the model may score, in both run modes
# GRCh38 primary-assembly chromosome lengths
CHROM_LENGTHS = {'1': 248956422, '2': 242193529, '3': 198295559, '4': 190214555, '5': 181538259,
                 '6': 170805979, '7': 159345973, '8': 145138636, '9': 138394717, '10': 133797422,
                 '11': 135086622, '12': 133275309, '13': 114364328, '14': 107043718, '15': 101991189,
                 '16': 90338345, '17': 83257441, '18': 80373285, '19': 58617616, '20': 64444167,
                 '21': 46709983, '22': 50818468, 'X': 156040895, 'Y': 57227415, 'M': 16569}
BUILD_FIELDS = ('build', 'assembly', 'genome_build')
SUPPORTED_BUILDS = {'GRCH38', 'HG38'}  # no build column means GRCh38 is assumed
GENE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._:|,;-]{0,63}$')  # one or more gene symbols, no free text
DNA_RUN = re.compile(r'[ACGTN]{32,}', re.I)  # sequence-like text; real annotation text has at most 11 in a row


def _err(code, field, message):
    return {'code': code, 'field': field, 'message': message}


def _get(record, key):
    """Field value, or None if absent, NaN, or blank."""
    v = record.get(key)
    if v is None or (isinstance(v, float) and math.isnan(v)) or (isinstance(v, str) and not v.strip()):
        return None
    return v


def _str(record, key):
    v = _get(record, key)
    return None if v is None else str(v).strip().upper()


def _int(record, key):
    """Whole-number field as int, or None if missing or not a whole number."""
    v = _get(record, key)
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return int(f) if f.is_integer() else None


def _chrom(record):
    """Chromosome without 'chr' prefix (accepts 'chr1' or '1'), or None."""
    c = _str(record, 'CHROM')
    return None if c is None else c.removeprefix('CHR')


def _alleles(record):
    """(REF, ALT) if both are valid ACGT strings, else None."""
    ref, alt = _str(record, 'REF'), _str(record, 'ALT')
    return (ref, alt) if ref and alt and ALLELE.match(ref) and ALLELE.match(alt) else None


def _window(record):
    """(wt, mut, 0-based variant index) if usable, else None."""
    wt, mut, sp = _str(record, 'wt_seq'), _str(record, 'mut_seq'), _int(record, 'start_pos')
    if wt is None or mut is None or sp is None or not 1 <= sp <= len(wt):
        return None
    return wt, mut, sp - 1


REQUIRED_FIELDS = ('CHROM', 'POS', 'REF', 'ALT', 'wt_seq', 'mut_seq', 'start_pos')

def check_required_fields(record):
    missing = [f for f in REQUIRED_FIELDS if _get(record, f) is None]
    if missing:
        return _err('MISSING_FIELD', ','.join(missing), f'missing or empty {missing}')


# ---- record checks ----


def check_chrom(record):
    c = _chrom(record)
    if c is not None and c not in VALID_CHROMS:
        return _err('UNKNOWN_CHROM', 'CHROM', f'unknown chromosome {_get(record, "CHROM")!r}')


def check_pos(record):
    if _get(record, 'POS') is None:
        return None
    pos = _int(record, 'POS')
    if pos is None or pos < 1:
        return _err('INVALID_POS', 'POS', f'POS must be a whole number >= 1, got {_get(record, "POS")!r}')


def check_pos_in_chrom(record):
    c, pos, ref = _chrom(record), _int(record, 'POS'), _str(record, 'REF')
    if c in CHROM_LENGTHS and pos is not None and pos >= 1:
        end = pos + len(ref or 'N') - 1
        if end > CHROM_LENGTHS[c]:
            return _err('POS_OUT_OF_RANGE', 'POS', f'variant ends at {end}, past the GRCh38 chr{c} end {CHROM_LENGTHS[c]}')


def check_build(record):
    for f in BUILD_FIELDS:
        b = _str(record, f)
        if b is not None and b not in SUPPORTED_BUILDS:
            return _err('UNSUPPORTED_BUILD', f, f'only GRCh38 is supported, got {_get(record, f)!r}')


def check_allele_alphabet(record):
    for f in ('REF', 'ALT'):
        a = _str(record, f)
        if a is not None and not ALLELE.match(a):
            return _err('INVALID_ALLELE', f, f'{f} must contain only A/C/G/T, got {a!r}')


def check_ref_ne_alt(record):
    ra = _alleles(record)
    if ra and ra[0] == ra[1]:
        return _err('NO_CHANGE', 'ALT', f'REF and ALT are identical ({ra[0]})')


def check_variant_type(record):
    ra = _alleles(record)
    if ra and len(ra[0]) != len(ra[1]):
        return _err('UNSUPPORTED_TYPE', 'ALT',
                    f'indels not supported: len(REF)={len(ra[0])} len(ALT)={len(ra[1])}')


# ---- sequence-window checks (precomputed wt_seq / mut_seq) ----

def check_seq_alphabet(record):
    for f in ('wt_seq', 'mut_seq'):
        s = _str(record, f)
        if s is not None and not SEQ.match(s):
            return _err('INVALID_SEQ', f, f'{f} must contain only A/C/G/T/N')


def check_seq_length(record):
    wt, mut = _str(record, 'wt_seq'), _str(record, 'mut_seq')
    if wt is not None and mut is not None and len(wt) != len(mut):
        return _err('LENGTH_MISMATCH', 'mut_seq', f'len(wt_seq)={len(wt)} != len(mut_seq)={len(mut)}')


def check_window_size(record):
    for f in ('wt_seq', 'mut_seq'):
        s = _str(record, f)
        if s is not None and len(s) > MAX_WINDOW:
            return _err('WINDOW_TOO_LARGE', f, f'{f} is {len(s)} bp, longer than the {MAX_WINDOW} bp window')


def check_start_pos(record):
    if _get(record, 'start_pos') is None:
        return None
    sp, wt, ref = _int(record, 'start_pos'), _str(record, 'wt_seq'), _str(record, 'REF')
    last = len(wt) - len(ref or 'N') + 1 if wt else None  # last start where the REF span still fits
    if sp is None or sp < 1 or (last is not None and sp > last):
        return _err('INVALID_START_POS', 'start_pos',
                    f'start_pos must be a whole number in 1..{last or "len(wt_seq)"}, '
                    f'got {_get(record, "start_pos")!r}')


def check_wt_mut_differ(record):
    wt, mut = _str(record, 'wt_seq'), _str(record, 'mut_seq')
    if wt is not None and wt == mut:
        return _err('SEQ_IDENTICAL', 'mut_seq', 'mut_seq is identical to wt_seq')


def check_ref_in_window(record):
    w, ref = _window(record), _str(record, 'REF')
    if w and ref:
        wt, _, i = w
        if wt[i:i + len(ref)] != ref:
            return _err('REF_MISMATCH', 'wt_seq',
                        f'wt_seq at start_pos is {wt[i:i + len(ref)]!r}, expected REF {ref!r}')


def check_alt_in_window(record):
    w, alt = _window(record), _str(record, 'ALT')
    if w and alt:
        _, mut, i = w
        if mut[i:i + len(alt)] != alt:
            return _err('ALT_MISMATCH', 'mut_seq',
                        f'mut_seq at start_pos is {mut[i:i + len(alt)]!r}, expected ALT {alt!r}')


def check_only_variant_differs(record):
    w, ref = _window(record), _str(record, 'REF')
    if w and ref and len(w[0]) == len(w[1]):
        wt, mut, i = w
        j = i + len(ref)
        if wt[:i] != mut[:i] or wt[j:] != mut[j:]:
            n = sum(a != b for a, b in zip(wt[:i] + wt[j:], mut[:i] + mut[j:]))
            return _err('EXTRA_CHANGES', 'mut_seq', f'{n} base(s) differ outside the variant')


# ---- annotation checks (optional ClinVar fields; skipped when absent) ----

def check_label(record):
    if _get(record, 'labels') is None:
        return None
    label = _int(record, 'labels')
    if label not in (0, 1):
        return _err('INVALID_LABEL', 'labels', f'labels must be 0 or 1, got {_get(record, "labels")!r}')
    sig = _get(record, 'CLNSIG')
    if sig is None:
        return None
    expected = 1 if sig in PATHOGENIC else 0 if sig in BENIGN else None
    if expected is None:
        return _err('UNMAPPED_SIGNIFICANCE', 'CLNSIG', f'no label mapping for {sig!r}')
    if label != expected:
        return _err('LABEL_MISMATCH', 'labels', f'labels={label} but CLNSIG={sig!r}')


def check_gene_name(record):
    g = _get(record, 'gene_name')
    if g is not None and not GENE.match(str(g).strip()):
        return _err('INVALID_GENE', 'gene_name', 'gene_name must be gene symbols only (letters, digits, . _ - : | , ;)')


def check_annotation_text(record):
    # annotations are copied into reports and messages that must never carry sequences
    for f in ('gene_name', 'cln_disease', 'CLNSIG'):
        v = _get(record, f)
        if v is not None and DNA_RUN.search(str(v)):
            return _err('SEQUENCE_IN_ANNOTATION', f, f'{f} contains sequence-like text')


def parse_hgvs(hgvs):
    """(chrom, pos, ref_len, alt) from a genomic HGVS string; alt is None for inversions."""
    if m := HGVS_SNV.match(hgvs):
        return m[1], int(m[2]), 1, m[4]
    if m := HGVS_DELINS.match(hgvs):
        return m[1], int(m[2]), int(m[3]) - int(m[2]) + 1, m[4]
    if m := HGVS_INV.match(hgvs):
        return m[1], int(m[2]), int(m[3]) - int(m[2]) + 1, None
    return None


def check_hgvs(record):
    hgvs = _get(record, 'cln_hgvs')
    ra, chrom, pos = _alleles(record), _chrom(record), _int(record, 'POS')
    if hgvs is None or ra is None or chrom is None or pos is None:
        return None
    parsed = parse_hgvs(str(hgvs).strip())
    if parsed is None:
        return _err('UNPARSED_HGVS', 'cln_hgvs', f'unsupported HGVS form {hgvs!r}')
    h_chrom, h_pos, h_ref_len, h_alt = parsed
    ref, alt = ra
    if h_alt is None:  # inversion: ALT is the reverse complement of REF
        h_alt = ref.translate(COMPLEMENT)[::-1]
    h_chrom = ACCESSION_CHROM.get(h_chrom, h_chrom)
    if (h_chrom, h_pos, h_ref_len, h_alt) != (chrom, pos, len(ref), alt):
        return _err('INCONSISTENT_HGVS', 'cln_hgvs', f'{hgvs!r} disagrees with CHROM/POS/REF/ALT')


def check_event_pos(record):
    if _get(record, 'event_pos') is None:
        return None
    ev, pos = _int(record, 'event_pos'), _int(record, 'POS')
    if ev is None:
        return _err('INVALID_EVENT_POS', 'event_pos',
                    f'event_pos must be a whole number, got {_get(record, "event_pos")!r}')
    if pos is not None and pos is not None and ev != pos:
        return _err('POS_MISMATCH', 'event_pos', f'event_pos={ev} != POS={pos}')


CHECKS = [check_required_fields, check_build,
          check_chrom, check_pos, check_pos_in_chrom, check_allele_alphabet, check_ref_ne_alt, check_variant_type,
          check_seq_alphabet, check_seq_length, check_window_size, check_start_pos, check_wt_mut_differ,
          check_ref_in_window, check_alt_in_window, check_only_variant_differs,
          check_label, check_hgvs, check_event_pos, check_gene_name, check_annotation_text]


def check_sequence_pair(pair, max_window):
    """Reason code if pair ({'wt_seq', 'mut_seq'}) is not one bounded, same-length, real variant window, else None."""
    if not isinstance(pair, dict) or not isinstance(pair.get('wt_seq'), str) or not isinstance(pair.get('mut_seq'), str):
        return 'BAD_SEQUENCE'
    if max(len(pair['wt_seq']), len(pair['mut_seq'])) > max_window:
        return 'WINDOW_TOO_LARGE'
    for check in (check_seq_alphabet, check_seq_length, check_wt_mut_differ):
        err = check(pair)
        if err:
            return err['code']
    return None


def validate_record(record):
    """Run every check and return all errors found; [] means valid."""
    return [e for check in CHECKS if (e := check(record))]


def validate_dataset(df):
    """Validate every row of a DataFrame; return {row index: errors} for rows that fail."""
    return {i: errors for i, r in tqdm(df.iterrows(), total=len(df), desc='Validating', unit='row')
            if (errors := validate_record(r))}


def run_validation(df, name):
    """Validate a dataset, print a per-dataset summary, and return {row index: errors} for failures."""
    failed = validate_dataset(df)
    for i, errors in failed.items():
        print(f'row {i} ID={df.at[i, "ID"] if "ID" in df else "-"}: {[e["code"] for e in errors]}')
    print(f'{name}: {len(df)} rows | passed={len(df) - len(failed)} failed={len(failed)}')
    codes = Counter(e['code'] for errors in failed.values() for e in errors)
    if codes:
        print('errors by code:', dict(codes))
    return failed
