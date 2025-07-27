import os
import numpy as np
import warnings
from collections import Counter
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from Bio import SeqIO
from sklearn import preprocessing
import pyrodigal

warnings.filterwarnings("ignore")

# === Feature extraction functions ===

def AAC_feature(sequence):
    AA = 'ACDEFGHIKLMNPQRSTVWY*'
    count = Counter(sequence)
    total = len(sequence)
    return [count.get(aa, 0) / total for aa in AA]

def physical_chemical_feature(sequence):
    seq_new = sequence.replace('*', '').replace('X','').replace('U','').replace('B','').replace('Z','').replace('J','')

    CE = 'CHONS'
    Chemi_stats = {
        'A': {'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 0},
        'C': {'C': 3, 'H': 7, 'O': 2, 'N': 1, 'S': 1},
        'D': {'C': 4, 'H': 7, 'O': 4, 'N': 1, 'S': 0},
        'E': {'C': 5, 'H': 9, 'O': 4, 'N': 1, 'S': 0},
        'F': {'C': 9, 'H': 11, 'O': 2, 'N': 1, 'S': 0},
        'G': {'C': 2, 'H': 5, 'O': 2, 'N': 1, 'S': 0},
        'H': {'C': 6, 'H': 9, 'O': 2, 'N': 3, 'S': 0},
        'I': {'C': 6, 'H': 13, 'O': 2, 'N': 1, 'S': 0},
        'K': {'C': 6, 'H': 14, 'O': 2, 'N': 2, 'S': 0},
        'L': {'C': 6, 'H': 13, 'O': 2, 'N': 1, 'S': 0},
        'M': {'C': 5, 'H': 11, 'O': 2, 'N': 1, 'S': 1},
        'N': {'C': 4, 'H': 8, 'O': 3, 'N': 2, 'S': 0},
        'P': {'C': 5, 'H': 9, 'O': 2, 'N': 1, 'S': 0},
        'Q': {'C': 5, 'H': 10, 'O': 3, 'N': 2, 'S': 0},
        'R': {'C': 6, 'H': 14, 'O': 2, 'N': 4, 'S': 0},
        'S': {'C': 3, 'H': 7, 'O': 3, 'N': 1, 'S': 0},
        'T': {'C': 4, 'H': 9, 'O': 3, 'N': 1, 'S': 0},
        'V': {'C': 5, 'H': 11, 'O': 2, 'N': 1, 'S': 0},
        'W': {'C': 11, 'H': 12, 'O': 2, 'N': 2, 'S': 0},
        'Y': {'C': 9, 'H': 11, 'O': 3, 'N': 1, 'S': 0}
    }
    count = Counter(seq_new)
    return [
        sum(Chemi_stats[aa][c] * count.get(aa, 0) for aa in count if aa in Chemi_stats)
        for c in CE
    ]

def molecular_weight(seq):
    seq_clean = seq.replace('*', '').replace('X','').replace('U','').replace('B','').replace('Z','').replace('J','')
    analysed_seq = ProteinAnalysis(seq_clean)
    analysed_seq.monoisotopic = True
    return [analysed_seq.molecular_weight()]

# === ORF prediction using Pyrodigal ===

def extract_protein_orfs_from_fasta(fasta_path, min_aa_length=30):
    with open(fasta_path) as f:
        genome_seq = f.read()
    gene_finder = pyrodigal.GeneFinder(meta=True)
    predictions = gene_finder.find_genes(genome_seq)
    proteins = [orf.translate() for orf in predictions if len(orf.translate()) >= min_aa_length]
    return proteins

# === Main processing ===
from tqdm import tqdm

def process_folder(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    fasta_files = [f for f in os.listdir(input_folder) if f.endswith(('.fasta', '.fa', '.fna'))]

    all_features = []
    names = []

    for fasta in tqdm(fasta_files, desc=f"Processing {input_folder}", unit="file"):
        path = os.path.join(input_folder, fasta)
        name = os.path.splitext(fasta)[0]

        try:
            orfs = extract_protein_orfs_from_fasta(path)
            features = []
            for seq in orfs:
                f = physical_chemical_feature(seq) + molecular_weight(seq) + AAC_feature(seq)
                features.append(f)
            if not features:
                tqdm.write(f"Warning: no valid ORFs in {name}")
                continue
            feats_np = np.array(features)
            summary = np.concatenate([
                feats_np.mean(axis=0),
                feats_np.max(axis=0),
                feats_np.min(axis=0),
                feats_np.std(axis=0),
                feats_np.var(axis=0),
                np.median(feats_np, axis=0)
            ])
            all_features.append(summary)
            names.append(name)
        except Exception as e:
            tqdm.write(f"Error processing {name}: {e}")

    # Normalize features
    scaler = preprocessing.MinMaxScaler()
    features_norm = scaler.fit_transform(all_features)

    # Save normalized features
    for i, name in enumerate(names):
        out_path = os.path.join(output_folder, f"{name}.txt")
        np.savetxt(out_path, features_norm[i])

if __name__ == '__main__':
    phage_folder = './Ordinal_Dataset/phages'
    host_folder = './Ordinal_Dataset/bacteria'

    process_folder(phage_folder, './oridnal_features/protein/phage')
    process_folder(host_folder, './oridnal_features/protein/bacteria')
