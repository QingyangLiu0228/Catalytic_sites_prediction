#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
data.py

All data-related code for Set A catalytic site prediction:
  - AA_3to1_dict constant
  - get_residue_center, get_pos_feature  (feature extraction)
  - ReadfromCSV                           (Dataset)
  - make_balanced_subset_from_csv         (1:k_neg balanced sampling)

Preserved verbatim from the original monolithic script except for renaming
and removing unused PDBDataset / random_rotation_matrix.
"""

import logging

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Subset

from Bio.PDB import PDBParser


# ============================================================================
# Constants
# ============================================================================
AA_3to1_dict = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}


# ============================================================================
# Integrity checks
# ============================================================================
def compare_pdb_esm_lengths(pdb_file, pdb_dir, esm_dir, esm_layer):
    """Validate that all PDB residue numbers fit within the ESM embedding.

    ESM is generated from the UniProt sequence (truncated at 1021 residues for
    ESM-2 3B). The PDB resid is 1-indexed and aligns with UniProt position.
    So the requirement is: max(PDB resid) <= ESM length.

    Returns True if the protein is usable, None otherwise (matches original
    contract so _filter_length_mismatches keeps working unchanged).
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file[:-3], pdb_dir + pdb_file)
    chain_id = pdb_file[5]
    chain = structure[0][chain_id]

    # --- Max PDB resid among standard residues ---
    max_pdb_resid = 0
    n_std = 0
    for residue in chain:
        het, resid, icode = residue.get_id()
        if het == ' ':
            if resid > max_pdb_resid:
                max_pdb_resid = resid
            n_std += 1

    # --- ESM length (must match get_pos_feature and on-disk names: sp|PDBID_CHAIN|esm.pt)
    esm_pt = "{}sp|{}_{}|esm.pt".format(esm_dir, pdb_file[:4], chain.get_id())
    rec = torch.load(esm_pt, map_location="cpu")
    if "representations" in rec:
        reps = rec["representations"]
    else:
        rec = next(iter(rec.values()))
        reps = rec["representations"]
    esm_len = reps[esm_layer].shape[0]

    ok = (max_pdb_resid <= esm_len)
    print("{}  n_std={}  max_resid={}  esm_len={}  ok={}".format(
        pdb_file, n_std, max_pdb_resid, esm_len, ok))

    if not ok:
        print("  PDB resid {} exceeds ESM length {}. Skipping.".format(
            max_pdb_resid, esm_len))
        return None

    return ok


# ============================================================================
# Feature extraction
# ============================================================================
def get_residue_center(residue):
    """Return the CA coordinate of a residue, or None if CA is missing."""
    # 1) try CA
    if "CA" in residue:
        return residue["CA"].get_coord()

    # 2) no reliable center
    else:
        return None


def get_pos_feature(pdb_file, box_size, voxel_size, pdb_dir, chain,
                    res_num, esm_dir, esm_layer):
    """Build a 3D voxel feature tensor centered on a target residue.

    The box contains (box_size/voxel_size)^3 voxels. Each voxel carries the
    ESM embedding of the neighboring residue whose CA falls into it.

    ESM is generated from the full UniProt sequence, so neighbor embeddings
    are looked up by (PDB resid - 1), which equals the UniProt 0-indexed
    position. This relies on PDB resid aligning with UniProt position,
    verified empirically on set A / set B.
    """
    numbox = int(box_size / voxel_size)
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_file[:-3], pdb_dir + pdb_file)

    # ---- Build PDB resid -> UniProt 0-indexed map for this chain ----
    chain_obj = structure[0][chain]
    residue_to_uniprot_index = {}
    for residue in chain_obj:
        het, resid, icode = residue.get_id()
        if het == ' ':  # standard residues only
            residue_to_uniprot_index[resid] = resid - 1

    # ---- Load ESM embeddings ----
    esm_file = "{}sp|{}_{}|esm.pt".format(esm_dir, pdb_file[:4], chain)
    try:
        esm_record = torch.load(esm_file, map_location="cpu")
        if "representations" in esm_record:
            esm_embeddings = esm_record["representations"][esm_layer]
        else:
            first_record = next(iter(esm_record.values()))
            esm_embeddings = first_record["representations"][esm_layer]
    except FileNotFoundError:
        logging.warning("ESM file not found: {}".format(esm_file))
        return np.zeros((numbox, numbox, numbox, 2560), dtype=np.float32)
    except KeyError:
        logging.error("ESM layer {} not found in {}".format(esm_layer, esm_file))
        return np.zeros((numbox, numbox, numbox, 2560), dtype=np.float32)

    embedding_dim = esm_embeddings.shape[1]
    esm_len = esm_embeddings.shape[0]

    # ---- Sanity-check the target residue ----
    residue = structure[0][chain][res_num]
    amino_acid_3 = residue.get_resname()
    if amino_acid_3 not in AA_3to1_dict:
        logging.warning("{}: {}{} is not a standard amino acid".format(
            pdb_file, res_num, amino_acid_3))

    residue_center = get_residue_center(residue)
    if residue_center is None:
        logging.warning("{}: {}{} does not have a reliable center".format(
            pdb_file, res_num, amino_acid_3))
        return np.zeros((numbox, numbox, numbox, embedding_dim), dtype=np.float32)

    # ---- Fill voxels with neighbor ESM embeddings ----
    struct_feature = np.zeros((numbox, numbox, numbox, embedding_dim),
                              dtype=np.float32)
    for residue in structure.get_residues():
        neighbor_residue_center = get_residue_center(residue)
        if neighbor_residue_center is None:
            continue

        delta_x = neighbor_residue_center[0] - residue_center[0]
        delta_y = neighbor_residue_center[1] - residue_center[1]
        delta_z = neighbor_residue_center[2] - residue_center[2]

        if abs(delta_x) < 10 and abs(delta_y) < 10 and abs(delta_z) < 10:
            _, neighbor_res_num, _ = residue.get_id()
            if neighbor_res_num in residue_to_uniprot_index:
                uniprot_idx = residue_to_uniprot_index[neighbor_res_num]
                # Guard against neighbors beyond ESM length (should not happen
                # after compare_pdb_esm_lengths filter, but keep safe).
                if 0 <= uniprot_idx < esm_len:
                    neighbor_embedding = esm_embeddings[uniprot_idx]

                    voxel_x = int(delta_x / voxel_size) + int(numbox / 2)
                    voxel_y = int(delta_y / voxel_size) + int(numbox / 2)
                    voxel_z = int(delta_z / voxel_size) + int(numbox / 2)
                    if (0 <= voxel_x < numbox
                            and 0 <= voxel_y < numbox
                            and 0 <= voxel_z < numbox):
                        struct_feature[voxel_x, voxel_y, voxel_z, :] = neighbor_embedding

    return struct_feature

# ============================================================================
# Dataset
# ============================================================================
class ReadfromCSV(Dataset):
    """Dataset loading training samples from a CSV index file.

    Each CSV row is a (pdb_file, chain, res_num, aa, label) tuple. The 3D
    voxel feature is computed lazily in __getitem__ by calling
    get_pos_feature.
    """

    def __init__(self, csv_path, pdb_dir, esm_dir, esm_layer=36,
                 box_size=20, voxel_size=1, debug_max_samples=None):
        self.csv_path = csv_path
        self.pdb_dir = pdb_dir
        self.esm_dir = esm_dir
        self.esm_layer = esm_layer
        self.box_size = box_size
        self.voxel_size = voxel_size

        self.df = pd.read_csv(csv_path)
        if debug_max_samples is not None:
            self.df = self.df.head(debug_max_samples)
            print("Debug mode: limited to {} samples".format(debug_max_samples))

        self._filter_length_mismatches()

        self.pos_num = (self.df['label'] == 1.0).sum()
        self.neg_num = (self.df['label'] == 0.0).sum()
        print("Loaded CSV: {} samples".format(len(self.df)))
        print("  Positive: {}".format(self.pos_num))
        print("  Negative: {}".format(self.neg_num))

    def _filter_length_mismatches(self):
        """Drop samples whose PDB chain length != ESM embedding length."""
        unique_pdbs = self.df['pdb_file'].unique()
        valid_pdbs = set()
        invalid_count = 0

        print("Validating PDB/ESM length matches for {} unique proteins...".format(
            len(unique_pdbs)))

        for pdb_file in unique_pdbs:
            try:
                is_valid = compare_pdb_esm_lengths(
                    pdb_file=pdb_file,
                    pdb_dir=self.pdb_dir,
                    esm_dir=self.esm_dir,
                    esm_layer=self.esm_layer,
                )
                if is_valid:
                    valid_pdbs.add(pdb_file)
                else:
                    invalid_count += 1
            except Exception as e:
                logging.warning("Error validating {}: {}".format(pdb_file, e))
                invalid_count += 1

        initial_count = len(self.df)
        self.df = self.df[self.df['pdb_file'].isin(valid_pdbs)]
        # After row drops, the index can keep original CSV row labels (gaps, large
        # max). Subset/iloc assume 0..len-1, so align label index with position.
        self.df = self.df.reset_index(drop=True)
        filtered_count = initial_count - len(self.df)

        print("Filtered {} samples from {} invalid proteins".format(
            filtered_count, invalid_count))
        print("Remaining: {} samples from {} valid proteins".format(
            len(self.df), len(valid_pdbs)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        pdb_file = row['pdb_file']
        chain = str(row['chain'])
        res_num = int(row['res_num'])
        label = float(row['label'])

        feat = get_pos_feature(
            pdb_file=pdb_file,
            box_size=self.box_size,
            voxel_size=self.voxel_size,
            pdb_dir=self.pdb_dir,
            chain=chain,
            res_num=res_num,
            esm_dir=self.esm_dir,
            esm_layer=self.esm_layer,
        )
        sample_id = "{}|{}|{}".format(pdb_file, chain, res_num)
        return feat, label, sample_id


# ============================================================================
# Balanced sampling
# ============================================================================
def make_balanced_subset_from_csv(dataset, k_neg, seed=42,
                                  replacement=False, shuffle=True):
    """Extract balanced positive/negative samples at 1:k_neg ratio.

    Args:
        dataset: Subset or ReadfromCSV instance
        k_neg: Number of negative samples per positive sample
        seed: Random seed
        replacement: Whether to sample negatives with replacement
        shuffle: Whether to shuffle the final indices

    Returns:
        torch.utils.data.Subset, or None if either class is empty.
    """
    if hasattr(dataset, 'dataset'):
        # Input is a Subset - pull labels from the underlying dataset
        base_dataset = dataset.dataset
        subset_indices = dataset.indices
        labels = base_dataset.df.iloc[subset_indices]['label'].to_numpy().astype(int)
    else:
        labels = dataset.df['label'].to_numpy().astype(int)
        subset_indices = list(range(len(labels)))

    pos_idx = [i for i, label in enumerate(labels) if label == 1]
    neg_idx = [i for i, label in enumerate(labels) if label == 0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return None

    rng = np.random.RandomState(seed)
    sampled, pool = [], neg_idx.copy()

    for i in pos_idx:
        sampled.append(i)
        if replacement:
            chosen = rng.choice(neg_idx, size=k_neg, replace=True).tolist()
        else:
            if len(pool) < k_neg:
                pool = neg_idx.copy()
            idxs = rng.choice(len(pool), size=k_neg, replace=False)
            chosen = [pool[j] for j in idxs]
            for j in sorted(idxs, reverse=True):
                pool.pop(j)
        sampled.extend(chosen)

    if shuffle:
        rng.shuffle(sampled)

    final_indices = [subset_indices[i] for i in sampled]

    print("Sampled {} indices: {} pos + {} neg (1:{})".format(
        len(sampled), len(pos_idx), len(pos_idx) * k_neg, k_neg))

    base_dataset = dataset.dataset if hasattr(dataset, 'dataset') else dataset
    return Subset(base_dataset, final_indices)
