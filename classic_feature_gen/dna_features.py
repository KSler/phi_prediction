import os
import numpy as np
import pandas as pd
from sklearn import preprocessing
import subprocess
from tqdm import tqdm

# Paths
phage_folder = './Ordinal_Dataset/phages'
host_folder = './Ordinal_Dataset/bacteria'
output_folder = './oridnal_features/dna'
os.makedirs(output_folder, exist_ok=True)

# in ilearn set k of Kmer and RCKmer to 3 instead of 2 (ilearn/descnucleotide/Kmer.py and RCKmer.py)
methods = ['Kmer','RCKmer','NAC','DNC','TNC','CKSNAP','PseEIIP']

def run_ilearn(fasta_path, output_csv, method):
    cmd = [
        'python', 'iLearn-nucleotide-basic.py',
        '--file', fasta_path,
        '--method', method,
        '--format', 'csv',
        '--out', output_csv
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error running iLearn for {fasta_path} with method {method}:")
        print(result.stderr)


def find_sequence_file(folder, id_):
    # Try .fna then .fasta
    for ext in ['.fna', '.fasta']:
        path = os.path.join(folder, f'{id_}{ext}')
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"No .fna or .fasta file found for {id_} in {folder}")

def process_group(group_name, folder):
    # Collect sequence IDs from both .fna and .fasta files
    ids = sorted([
        os.path.splitext(f)[0]
        for f in os.listdir(folder)
        if f.endswith(('.fna', '.fasta'))
    ])
    features_by_method = []


    for method in tqdm(methods, desc=f'Running iLearn methods for {group_name}'):
        method_output = os.path.join(output_folder, f'{group_name}_{method}.csv')

        # Combine all sequences into one temporary file
        combined_fasta = os.path.join(folder, f'temp_{group_name}.fasta')
        with open(combined_fasta, 'w') as outfile:
            for id_ in ids:
                filepath = find_sequence_file(folder, id_)
                with open(filepath) as infile:
                    outfile.write(infile.read())

        run_ilearn(combined_fasta, method_output, method)

        df = pd.read_csv(method_output, header=None)
        data = df.iloc[:, 1:].values  # Skip the first column
        features_by_method.append(data)

        os.remove(combined_fasta)  # clean up temp file

    combined_features = np.hstack(features_by_method)
    return ids, combined_features

# --- Run for both groups ---
phage_ids, phage_features = process_group('phage', phage_folder)
host_ids, host_features = process_group('host', host_folder)

# --- Normalize ---
phage_features_norm = preprocessing.MinMaxScaler().fit_transform(phage_features)
host_features_norm = preprocessing.MinMaxScaler().fit_transform(host_features)

# --- Save ---
phage_outdir = './dna_features_ordinal_data/phage_dna_norm_features/'
host_outdir = './dna_features_ordinal_data/host_dna_norm_features/'
os.makedirs(phage_outdir, exist_ok=True)
os.makedirs(host_outdir, exist_ok=True)

print("Saving phage features...")
for i in tqdm(range(len(phage_ids)), desc='Saving phage features'):
    np.savetxt(f'{phage_outdir}{phage_ids[i]}.txt', phage_features_norm[i, :])

print("Saving host features...")
for i in tqdm(range(len(host_ids)), desc='Saving host features'):
    np.savetxt(f'{host_outdir}{host_ids[i]}.txt', host_features_norm[i, :])

