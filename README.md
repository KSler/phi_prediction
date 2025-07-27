This notebook performs classification and regression tasks using both classical features and LLM-derived embeddings to predict phage-host interaction strength. Two models are evaluated: XGBoost and Random Forest.

LLM-based features are derived following the approach from PhageHostLearn:
https://github.com/dimiboeckaerts/PhageHostLearn?tab=readme-ov-file

Classical features are based on methods from the PHIAF study,
they can be computed using ILearn in combination with the dna_features.py and the protein_feature.py included in this Notebook.


Data Availability:

Escherichia Dataset:
    Bacteria genomes:
    https://figshare.com/articles/dataset/Genome_assembly_of_the_Escherichia_Picard_collection/25941691/1
    Phage genomes:
    https://github.com/mdmparis/coli_phage_interactions_2023/tree/main/data/genomics/phages/FNA

Klebsiella Dataset
    Phage and Bacteria genomes:
    https://zenodo.org/records/11061100
