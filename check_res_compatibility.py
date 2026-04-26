#!/usr/bin/env python
# check_pdb_fasta_alignment.py
"""
For every unique protein in the CSV, check that each PDB standard residue
(het==' ') matches FASTA[resid-1] (1-based resid → 0-based FASTA index).

Supports two naming styles:
- setA-style: pdb_file = ``1al6_A.pdb``, FASTA header ``>1al6_A`` (stem = PDB+chain in filename)
- setB-style: pdb_file = ``A0A084JZF2.pdb``, FASTA header ``>A0A084JZF2`` (UniProt accession; chain
  comes from the CSV ``chain`` column, not the filename)

Uses ``Path(pdb_file).stem`` as the primary FASTA lookup key, and the CSV ``chain`` column
for the Bio.PDB chain id.
"""
import re
from pathlib import Path

import pandas as pd
from Bio.PDB import PDBParser
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore")

# ============================================================================
# Config - EDIT THESE
# ============================================================================
CSV = "/data/data3/conglab/s441865/dataset/setB_test.csv"
# Trailing slash optional — paths are built with Path(PDB_DIR) / pdb_file
PDB_DIR = "/data/data1/conglab/qcong/for/rongqing/prepare_uniprot_training/step17_pdbs"

# Option A: one combined FASTA (setA: >1al6_A, setB: >A0A084JZF2)
FASTA_FILE = "/data/data1/conglab/qcong/for/rongqing/prepare_uniprot_training/step14_sample_seqs.fa"
# Option B: directory of per-protein .fa (only used if key missing from combined)
FASTA_DIR = None  # e.g. "/data/data3/conglab/s441865/prepare_mcat_training/step4/"

assert FASTA_FILE or FASTA_DIR, "Set FASTA_FILE and/or FASTA_DIR"

# If True, print one line for every protein (very verbose for ~900 structures)
VERBOSE = False
# How many example mismatches to show per failed protein in the failure report
MISMATCH_EXAMPLES = 3

# ============================================================================
AA_3to1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


def load_fasta_single_file(path):
    """Parse a combined fasta file. Returns {header_id: sequence}."""
    seqs = {}
    current_id = None
    current_seq = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current_id is not None:
                    seqs[current_id] = "".join(current_seq)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                current_seq.append(line)
    if current_id is not None:
        seqs[current_id] = "".join(current_seq)
    return seqs


def load_fasta_per_protein(fasta_dir, stem_id, chain_id):
    """Try to find a per-protein fasta file. Returns sequence or None."""
    candidates = [
        Path(fasta_dir) / f"sp|{stem_id}_{chain_id}.fasta",
        Path(fasta_dir) / f"sp|{stem_id}_{chain_id}.fa",
        Path(fasta_dir) / f"{stem_id}_{chain_id}.fasta",
        Path(fasta_dir) / f"{stem_id}.fasta",
        Path(fasta_dir) / f"{stem_id}.fa",
    ]
    for p in candidates:
        if p.exists():
            lines = p.read_text().strip().split("\n")
            seq = "".join(l for l in lines if not l.startswith(">"))
            return seq
    return None


def get_fasta_for(fasta_stem, chain_id, combined_seqs):
    """Look up the fasta sequence by header id. ``fasta_stem`` = Path(pdb_file).stem."""
    chain_id = str(chain_id).strip()
    keys_to_try = [
        fasta_stem,
        f"{fasta_stem}_{chain_id}",
        f"sp|{fasta_stem}_{chain_id}",
        f"sp|{fasta_stem}|{chain_id}",
    ]
    # PDB-style 4-letter code in filename without chain (e.g. ``1al6.pdb`` + chain A -> 1al6_A)
    if "_" not in fasta_stem and len(fasta_stem) >= 4:
        pdb4 = fasta_stem[:4]
        if re.match(r"^[0-9][A-Za-z0-9]{3}$", pdb4):
            keys_to_try.append(f"{pdb4}_{chain_id}")

    for key in keys_to_try:
        if key in combined_seqs:
            return combined_seqs[key]

    for k in combined_seqs:
        if k == fasta_stem or k.startswith(fasta_stem + "_"):
            return combined_seqs[k]

    if FASTA_DIR:
        return load_fasta_per_protein(FASTA_DIR, fasta_stem, chain_id)
    return None


def compare_chain_to_fasta(chain, fasta_seq):
    """
    For every standard residue, compare AA to fasta[resid-1].
    Returns (n_checked, n_match, n_mismatch, n_out_of_range, first_examples)
    first_examples: list of (resid, icode, pdb_aa, fasta_slot, note) for mismatches
    """
    n_match = 0
    n_mismatch = 0
    n_oor = 0
    examples = []
    n_checked = 0

    for residue in chain:
        het, resid, icode = residue.get_id()
        if het != " ":
            continue
        pdb_aa = AA_3to1.get(residue.get_resname(), "?")
        idx = resid - 1
        n_checked += 1
        if idx < 0 or idx >= len(fasta_seq):
            n_oor += 1
            n_mismatch += 1
            if len(examples) < MISMATCH_EXAMPLES:
                examples.append(
                    (resid, icode, pdb_aa, idx, f"out of range (len={len(fasta_seq)})")
                )
        else:
            fasta_aa = fasta_seq[idx]
            if pdb_aa == fasta_aa:
                n_match += 1
            else:
                n_mismatch += 1
                if len(examples) < MISMATCH_EXAMPLES:
                    examples.append((resid, icode, pdb_aa, idx, f"fasta[{idx}]={fasta_aa}"))

    return n_checked, n_match, n_mismatch, n_oor, examples


# ============================================================================
# Main
# ============================================================================
def main():
    df = pd.read_csv(CSV)
    unique_pdbs = sorted(df["pdb_file"].unique().tolist())
    n_total = len(unique_pdbs)

    combined_seqs = {}
    if FASTA_FILE:
        print(f"Loading combined fasta from {FASTA_FILE} ...")
        combined_seqs = load_fasta_single_file(FASTA_FILE)
        print(f"  loaded {len(combined_seqs)} sequences\n")

    parser = PDBParser(QUIET=True)

    protein_ok = 0
    protein_fail = 0
    protein_no_fasta = 0
    pdb_load_fail = 0

    total_residues = 0
    total_match = 0
    total_mismatch = 0
    total_oor = 0

    failed_details = []  # (pdb_file, n_checked, n_mismatch, n_oor, examples, fasta_len)

    for pdb_file in tqdm(unique_pdbs, desc="PDB + FASTA check", unit="pdb"):
        sub = df.loc[df["pdb_file"] == pdb_file, "chain"]
        chain_id = str(sub.iloc[0]).strip()
        fasta_stem = Path(pdb_file).stem
        pdb_path = Path(PDB_DIR) / pdb_file

        try:
            structure = parser.get_structure("x", str(pdb_path))
            chain = structure[0][chain_id]
        except Exception as e:
            pdb_load_fail += 1
            if VERBOSE:
                print(f"[{pdb_file}] PDB load failed: {e}")
            continue

        fasta_seq = get_fasta_for(fasta_stem, chain_id, combined_seqs)
        if fasta_seq is None:
            protein_no_fasta += 1
            if VERBOSE:
                print(f"[{pdb_file}] FASTA not found, skipping")
            continue

        n_checked, n_match, n_mismatch, n_oor, examples = compare_chain_to_fasta(
            chain, fasta_seq
        )
        total_residues += n_checked
        total_match += n_match
        total_mismatch += n_mismatch
        total_oor += n_oor

        aligned = n_mismatch == 0
        if aligned:
            protein_ok += 1
        else:
            protein_fail += 1
            failed_details.append(
                (pdb_file, n_checked, n_match, n_mismatch, n_oor, examples, len(fasta_seq))
            )

        if VERBOSE:
            status = "OK  " if aligned else "FAIL"
            print(
                f"[{pdb_file}] {status}  residues={n_checked}  "
                f"match={n_match}  mismatch={n_mismatch}  fasta_len={len(fasta_seq)}"
            )
            for row in examples:
                print(f"    resid {row[0]}{row[1].strip()}: {row[4]}")

    print("\n" + "=" * 60)
    print(f"Summary — unique proteins: {n_total}")
    print("=" * 60)
    print(f"Proteins — fully consistent (all residues) : {protein_ok}")
    print(f"Proteins — at least one resid mismatch     : {protein_fail}")
    print(f"Proteins — FASTA not found                 : {protein_no_fasta}")
    print(f"Proteins — PDB load failed                 : {pdb_load_fail}")
    if protein_ok + protein_fail > 0:
        pct = 100.0 * protein_ok / (protein_ok + protein_fail)
        print(f"Protein pass rate (of those with FASTA)  : {pct:.1f}%")
    print("-" * 60)
    print(f"Residues checked (std AA, all proteins)  : {total_residues}")
    print(f"Residue positions matching               : {total_match}")
    print(f"Residue positions mismatching (incl. OOR): {total_mismatch}")
    print(f"  of which out-of-range (FASTA too short) : {total_oor}")
    if total_residues > 0:
        r_pct = 100.0 * total_match / total_residues
        print(f"Residue match rate                        : {r_pct:.2f}%")

    if failed_details and not VERBOSE:
        print("\n" + "-" * 60)
        print(f"Failed proteins (first {min(len(failed_details), 40)} of {len(failed_details)}); "
              f"up to {MISMATCH_EXAMPLES} example rows each:")
        for item in failed_details[:40]:
            pdb_file, n_checked, n_match, n_mismatch, n_oor, examples, flen = item
            print(
                f"  {pdb_file}  checked={n_checked}  match={n_match}  "
                f"mismatch={n_mismatch}  OOR={n_oor}  fasta_len={flen}"
            )
            for row in examples:
                print(f"      resid {row[0]}{row[1] if row[1] and str(row[1]).strip() else ''}  {row[4]}")
        if len(failed_details) > 40:
            print(f"  ... and {len(failed_details) - 40} more failed proteins (set VERBOSE=True for all).")


if __name__ == "__main__":
    main()
