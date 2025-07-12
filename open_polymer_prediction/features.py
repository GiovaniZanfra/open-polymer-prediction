from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys, rdMolDescriptors

# Paths
datair = Path("data/interim")
PROCESSED = Path("data/processed")

# SMARTS patterns for fragment counts
SMARTS_PATTERNS = {
    "oh": "[OX2H]",
    "nh2": "[NX3;H2]",
    "cooh": "C(=O)[OX2H1]",
    "aromatic_ring": "a1aaaaa1",
    "halogen": "[F,Cl,Br,I]",
    "nitro": "[NX3](=O)=O",
}


def compute_2d_descriptors(mol):
    """Compute selected 2D molecular descriptors"""
    feats = {}
    # weights and sizes
    feats['MolWt'] = Descriptors.MolWt(mol)
    feats['ExactMolWt'] = Descriptors.ExactMolWt(mol)
    feats['HeavyAtomMolWt'] = Descriptors.HeavyAtomMolWt(mol)
    feats['HeavyAtomCount'] = Descriptors.HeavyAtomCount(mol)
    feats['NumValenceElectrons'] = Descriptors.NumValenceElectrons(mol)
    feats['NumRadicalElectrons'] = Descriptors.NumRadicalElectrons(mol)
    feats['NumHDonors'] = Descriptors.NumHDonors(mol)
    feats['NumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    # logP etc
    feats['MolLogP'] = Descriptors.MolLogP(mol)
    feats['MolMR'] = Descriptors.MolMR(mol)
    # TPSA
    feats['TPSA'] = Descriptors.TPSA(mol)
    # topological indices
    for name in ['Chi0','Chi1']:
        feats[name] = getattr(Descriptors, name)(mol)
    for k in [1,2,3]:
        feats[f'Kappa{k}'] = getattr(Descriptors, f'Kappa{k}')(mol)
    # E-state & BCUT
    feats['MaxEStateIndex'] = Descriptors.MaxEStateIndex(mol)
    feats['MinEStateIndex'] = Descriptors.MinEStateIndex(mol)
    feats['BalabanJ'] = Descriptors.BalabanJ(mol)
    feats['BertzCT'] = Descriptors.BertzCT(mol)
    for suffix in ['MWHI','MWLOW','CHGHI','CHGLO','LOGPHI','LOGPLOW','MRHI','MRLOW']:
        feats[f'BCUT2D_{suffix}'] = getattr(Descriptors, f'BCUT2D_{suffix}')(mol)
    # VSA
    for pref in ['SlogP_VSA','SMR_VSA','PEOE_VSA','VSA_EState']:
        for i in range(9):
            key = f'{pref}{i}'
            if hasattr(Descriptors, key): feats[key] = getattr(Descriptors, key)(mol)
    # lipinski / MolSurf
    feats['NumRotatableBonds'] = Descriptors.NumRotatableBonds(mol)
    feats['NumBridgeheadAtoms'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    feats['NumHeteroatoms'] = rdMolDescriptors.CalcNumHeteroatoms(mol)
    feats['FractionCSP3'] = Descriptors.FractionCSP3(mol)
    feats['RingCount'] = Descriptors.RingCount(mol)
    feats['NumAromaticRings'] = Descriptors.NumAromaticRings(mol)
    feats['NumAliphaticRings'] = Descriptors.NumAliphaticRings(mol)
    feats['NumSaturatedRings'] = Descriptors.NumSaturatedRings(mol)
    # MolFormula length
    feats['MolFormula'] = len(rdMolDescriptors.CalcMolFormula(mol))
    return feats


def compute_simple_counts(mol):
    feats = {}
    n_atoms = mol.GetNumAtoms()
    feats['AtomCount'] = n_atoms
    feats['BondCount'] = mol.GetNumBonds()
    feats['PercentAromaticAtoms'] = sum(a.GetIsAromatic() for a in mol.GetAtoms()) / n_atoms
    # bond types
    types = [b.GetBondType() for b in mol.GetBonds()]
    feats['DoubleBondFrac'] = types.count(Chem.BondType.DOUBLE) / len(types) if types else 0
    feats['TripleBondFrac'] = types.count(Chem.BondType.TRIPLE) / len(types) if types else 0
    # rings
    ring_info = mol.GetRingInfo()
    feats['NumRings'] = ring_info.NumRings()
    feats['NumAtomsInRings'] = sum(len(r) for r in ring_info.AtomRings())
    return feats


def compute_fingerprints(mol):
    feats = {}
    # Morgan
    mfp = AllChem.GetMorganFingerprintAsBitVect(mol,2,1024)
    for i, bit in enumerate(mfp): feats[f'MorganBit{i}'] = int(bit)
    # MACCS
    maccs = MACCSkeys.GenMACCSKeys(mol)
    for i, bit in enumerate(maccs): feats[f'MACCS{i}'] = int(bit)
    # atom pairs + torsion
    ap = AllChem.GetHashedAtomPairFingerprintAsBitVect(mol)
    feats['AtomPair_fp_sum'] = sum(ap)
    tor = AllChem.GetHashedTopologicalTorsionFingerprintAsBitVect(mol)
    feats['Torsion_fp_sum'] = sum(tor)
    return feats


def compute_fragment_counts(mol):
    feats = {}
    for name, smarts in SMARTS_PATTERNS.items():
        patt = Chem.MolFromSmarts(smarts)
        feats[f'Count_{name}'] = len(mol.GetSubstructMatches(patt))
    # Lipinski
    feats['LipinskiNumHDonors'] = Descriptors.NumHDonors(mol)
    feats['LipinskiNumHAcceptors'] = Descriptors.NumHAcceptors(mol)
    feats['LipinskiNumAliphaticCarbocycles'] = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
    feats['LipinskiNumAromaticCarbocycles'] = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
    return feats


def compute_graph_features(mol):
    feats = {}
    adj = Chem.rdmolops.GetAdjacencyMatrix(mol)
    G = nx.from_numpy_array(adj)
    feats['GraphDiameter'] = nx.diameter(G) if nx.is_connected(G) else 0
    feats['GraphAvgPath'] = nx.average_shortest_path_length(G) if nx.is_connected(G) else 0
    feats['GraphCycles'] = len(list(nx.cycle_basis(G)))
    feats['GraphClusteringAvg'] = np.mean(list(nx.clustering(G).values()))
    return feats


def featurize_df(df):
    all_feats = []
    total = len(df)
    for i, smi in enumerate(df['SMILES']):
        print(f"{i}/{total}")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            all_feats.append({})
            continue
        feats = {}
        feats.update(compute_2d_descriptors(mol))
        feats.update(compute_simple_counts(mol))
        feats.update(compute_fingerprints(mol))
        feats.update(compute_fragment_counts(mol))
        feats.update(compute_graph_features(mol))
        all_feats.append(feats)
    return pd.DataFrame(all_feats)


def main():
    PROCESSED.mkdir(exist_ok=True)
    for split in ['train','test']:
        df = pd.read_parquet(datair/f"{split}.parquet")
        feats = featurize_df(df)
        out = pd.concat([df.reset_index(drop=True), feats], axis=1)
        out.to_parquet(PROCESSED/f"{split}.parquet", index=False)

if __name__ == '__main__':
    main()
