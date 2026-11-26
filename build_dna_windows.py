#!/usr/bin/env python3
"""Generate centered wild-type/mutant DNA windows for a variant CSV with the ClinVar CSV columns.

Examples
--------
# Evo2 windows (variant begins at position 4097):
python build_dna_windows.py --seq-len 8192

# 200 bp on each side (401 bp for an SNV; variant begins at position 201):
python build_dna_windows.py --flank-bp 200

# The older convention of a 200 bp total window:
python build_dna_windows.py --seq-len 200
"""

from __future__ import annotations

import argparse
import csv
import gzip
from pathlib import Path
from typing import TextIO

from pyfaidx import Fasta


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = Path("dataset/clinvar_cm_rare_missense_dna_8192.csv")
DEFAULT_FASTA = PROJECT_DIR / "data" / "hg38.fa"

# the column names of dataset/clinvar_cm_rare_missense_dna_8192.csv
REQUIRED_COLUMNS = {
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "gene_name",
    "ID",
    "cln_significance",
}

METADATA_COLUMNS = [
    "CHROM",
    "POS",
    "REF",
    "ALT",
    "gene_name",
    "ID",
    "cln_significance",
    "cln_disease",
    "cln_hgvs",
    "max_af_clinvar",
]


def normalize_chromosome(chromosome: str) -> str:
    chromosome = chromosome.strip()
    return chromosome if chromosome.startswith("chr") else f"chr{chromosome}"


def normalize_variant(ref: str, alt: str, position: int) -> tuple[int, str, str]:
    """Trim shared allele bases and return the first changed hg38 position."""
    ref = ref.upper()
    alt = alt.upper()

    while ref and alt and ref[0] == alt[0]:
        ref = ref[1:]
        alt = alt[1:]
        position += 1

    while ref and alt and ref[-1] == alt[-1]:
        ref = ref[:-1]
        alt = alt[:-1]

    return position, ref, alt


def significance_label(row: dict[str, str]) -> str:
    """ClinVar significance text; empty, conflicting or uncertain becomes Uncertain."""
    raw = row.get("cln_significance", "").strip()
    text = raw.replace("_", " ").strip()
    lower = text.lower()

    if "conflicting" in lower or "uncertain" in lower or not text:
        return "Uncertain"
    if "benign" in lower:
        return text
    if "pathogenic" in lower:
        return text
    return "Uncertain"


def label_group(label: str) -> str:
    lower = label.lower()
    if "uncertain" in lower or "conflicting" in lower:
        return "vus"
    if "pathogenic" in lower:
        return "pathogenic"
    if "benign" in lower:
        return "benign"
    return "vus"


def fetch(fasta: Fasta, chrom: str, start_1based: int, length: int) -> str:
    if length <= 0:
        return ""
    if start_1based < 1:
        raise ValueError(f"Window begins before {chrom}:1")
    return str(
        fasta[chrom][start_1based - 1 : start_1based - 1 + length]
    ).upper()


def build_fixed_length_windows(
    fasta: Fasta,
    chrom: str,
    position: int,
    ref: str,
    alt: str,
    sequence_length: int,
) -> tuple[str, str, int]:
    """Build equal-length sequences with the changed event centered.

    For indels, each allele receives the amount of downstream reference
    sequence needed to keep both outputs exactly ``sequence_length`` bases.
    """
    event_position, ref_event, alt_event = normalize_variant(ref, alt, position)
    start_pos = sequence_length // 2 + 1
    left_length = start_pos - 1
    left = fetch(fasta, chrom, event_position - left_length, left_length)

    observed_ref = fetch(fasta, chrom, event_position, len(ref_event))
    if observed_ref != ref_event:
        raise ValueError(
            f"REF mismatch at {chrom}:{event_position}: "
            f"CSV={ref_event!r}, hg38={observed_ref!r}"
        )

    right_start = event_position + len(ref_event)
    wt_right_length = sequence_length - left_length - len(ref_event)
    mut_right_length = sequence_length - left_length - len(alt_event)
    if wt_right_length < 0 or mut_right_length < 0:
        raise ValueError("Allele is longer than the requested sequence window")

    wt_seq = left + ref_event + fetch(fasta, chrom, right_start, wt_right_length)
    mut_seq = left + alt_event + fetch(fasta, chrom, right_start, mut_right_length)

    if len(wt_seq) != sequence_length or len(mut_seq) != sequence_length:
        raise ValueError(
            f"Could not build full {sequence_length}-bp window at "
            f"{chrom}:{position}"
        )
    return wt_seq, mut_seq, start_pos


def build_flanked_windows(
    fasta: Fasta,
    chrom: str,
    position: int,
    ref: str,
    alt: str,
    flank_bp: int,
) -> tuple[str, str, int]:
    """Build sequences containing each complete allele plus equal flanks."""
    event_position, ref_event, alt_event = normalize_variant(ref, alt, position)
    left = fetch(fasta, chrom, event_position - flank_bp, flank_bp)

    observed_ref = fetch(fasta, chrom, event_position, len(ref_event))
    if observed_ref != ref_event:
        raise ValueError(
            f"REF mismatch at {chrom}:{event_position}: "
            f"CSV={ref_event!r}, hg38={observed_ref!r}"
        )

    right_start = event_position + len(ref_event)
    right = fetch(fasta, chrom, right_start, flank_bp)
    return left + ref_event + right, left + alt_event + right, flank_bp + 1


def open_output(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        return gzip.open(path, "wt", newline="")
    return path.open("w", newline="")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--reference-fasta", type=Path, default=DEFAULT_FASTA)
    length = parser.add_mutually_exclusive_group()
    length.add_argument(
        "--seq-len",
        type=int,
        default=None,
        help="Total fixed sequence length (default: 8192 for Evo2)",
    )
    length.add_argument(
        "--flank-bp",
        type=int,
        help="Bases on each side; total SNV length is 2*flank+1",
    )
    parser.add_argument(
        "--subset",
        choices=["all", "vus", "non-vus"],
        default="all",
        help="Rows to export (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .csv or .csv.gz; defaults to a name based on length/subset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for testing",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sequence_length = (
        2 * args.flank_bp + 1 if args.flank_bp is not None else args.seq_len or 8192
    )
    if sequence_length < 1:
        raise ValueError("Sequence length must be positive")

    if args.output is None:
        suffix = f"flank_{args.flank_bp}" if args.flank_bp is not None else str(sequence_length)
        args.output = (
            PROJECT_DIR
            / "data"
            / f"VUS_64_genes_rating_dna_{suffix}_{args.subset}.csv.gz"
        )

    if not args.input.is_file():
        raise FileNotFoundError(args.input)
    if not args.reference_fasta.is_file():
        raise FileNotFoundError(args.reference_fasta)

    fasta = Fasta(
        str(args.reference_fasta), as_raw=True, sequence_always_upper=True
    )
    output_columns = ["wt_seq", "mut_seq", "start_pos", "labels"] + METADATA_COLUMNS
    processed = written = 0
    group_counts = {"vus": 0, "benign": 0, "pathogenic": 0}

    with args.input.open(newline="") as source, open_output(args.output) as destination:
        reader = csv.DictReader(source)
        missing = REQUIRED_COLUMNS.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required columns: {sorted(missing)}")

        writer = csv.DictWriter(destination, fieldnames=output_columns)
        writer.writeheader()

        for row in reader:
            if args.limit is not None and processed >= args.limit:
                break
            processed += 1

            label = significance_label(row)
            group = label_group(label)
            if args.subset == "vus" and group != "vus":
                continue
            if args.subset == "non-vus" and group == "vus":
                continue

            chrom = normalize_chromosome(row["CHROM"])
            if args.flank_bp is not None:
                wt_seq, mut_seq, start_pos = build_flanked_windows(
                    fasta=fasta,
                    chrom=chrom,
                    position=int(row["POS"]),
                    ref=row["REF"],
                    alt=row["ALT"],
                    flank_bp=args.flank_bp,
                )
            else:
                wt_seq, mut_seq, start_pos = build_fixed_length_windows(
                    fasta=fasta,
                    chrom=chrom,
                    position=int(row["POS"]),
                    ref=row["REF"],
                    alt=row["ALT"],
                    sequence_length=sequence_length,
                )

            output_label: str | int = label
            if args.subset == "non-vus":
                output_label = 1 if group == "pathogenic" else 0

            output_row = {
                "wt_seq": wt_seq,
                "mut_seq": mut_seq,
                "start_pos": start_pos,
                "labels": output_label,
            }
            output_row.update({column: row.get(column, "") for column in METADATA_COLUMNS})
            writer.writerow(output_row)
            written += 1
            group_counts[group] += 1

    print(f"Input rows read: {processed:,}")
    print(f"Rows written: {written:,}")
    print(f"Label groups written: {group_counts}")
    if args.flank_bp is not None:
        print(
            f"Sequence length: {2 * args.flank_bp:,} bp plus complete allele "
            "(variable for indels)"
        )
        print(f"Variant start position: {args.flank_bp + 1:,} (1-based)")
    else:
        print(f"Sequence length: {sequence_length:,} bp")
        print(f"Variant start position: {sequence_length // 2 + 1:,} (1-based)")
    print(f"Output: {args.output}")


if __name__ == "__main__":
    main()
