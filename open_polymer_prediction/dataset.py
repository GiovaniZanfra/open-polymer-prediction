from pathlib import Path

import pandas as pd
from rdkit import Chem

# Paths
RAW = Path("data/raw")
INTERIM = Path("data/interim")
EXTERNAL = Path("data/external")


def make_smile_canonical(smile):
    """Canonicalize a SMILES string or return None if invalid"""
    try:
        mol = Chem.MolFromSmiles(smile)
        return Chem.MolToSmiles(mol, canonical=True)
    except:
        return None

def add_extra_data(df: pd.DataFrame, df_extra: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Merge extra dataset into train by SMILES. Impute existing and append new unique SMILES.

    Parameters:
    - df: original training DataFrame with columns ['SMILES', ..., target]
    - df_extra: DataFrame with columns ['SMILES', value]
    - target: name of the target column ('Tc', 'Tg', 'Density' etc.)

    Returns:
    - df augmented with extra samples and imputed missing
    """
    before = df[target].notna().sum()
    # rename extra column
    df_extra = df_extra.rename(columns={df_extra.columns[1]: target})
        # ensure numeric target for aggregation
    df_extra[target] = pd.to_numeric(df_extra.iloc[:,1], errors='coerce')
    # canonicalize
    df_extra['SMILES'] = df_extra['SMILES'].map(make_smile_canonical)
    df_extra = df_extra.dropna(subset=['SMILES', target])
    # aggregate duplicates
    df_extra = df_extra.groupby('SMILES', as_index=False)[target].mean()

    # prepare mapping for imputation
    mapping = df_extra.set_index('SMILES')[target].to_dict()

    # identify existing missing entries to impute
    mask = df['SMILES'].isin(mapping.keys()) & df[target].isna()
    # vectorized assignment without dropping columns
    df.loc[mask, target] = df.loc[mask, 'SMILES'].map(mapping)
    # mark imputed as external
    df.loc[mask, 'is_external'] = True

    # append entirely new rows
    new_smiles = set(mapping.keys()) - set(df['SMILES'])
    df_new = pd.DataFrame({
        'SMILES': list(new_smiles),
        target: [mapping[s] for s in new_smiles]
    })

    # create empty columns for df_new to match df columns
    for col in df.columns:
        if col not in df_new.columns:
            df_new[col] = pd.NA

    df = pd.concat([df, df_new[df.columns]], ignore_index=True)

    after = df[target].notna().sum()
    print(f"Added {after - before} samples for target '{target}' (new SMILES: {len(new_smiles)})")
    return df


def process_raw():
    # 1. Load raw data
    train = pd.read_csv(RAW / 'train.csv')
    test = pd.read_csv(RAW / 'test.csv')

    # flag original data
    train['is_external'] = False
    test['is_external'] = False

    # canonicalize SMILES in train
    train['SMILES'] = train['SMILES'].map(make_smile_canonical)

    # 2. Load and merge external datasets
    # Thermal conductivity (Tc)
    tc_df = pd.read_csv(EXTERNAL / 'tc_smiles' / 'Tc_SMILES.csv')
    train = add_extra_data(train, tc_df[['SMILES', 'TC_mean']], 'Tc')

    # Glass transition (Tg) from two sources
    tg2_df = pd.read_csv(EXTERNAL / 'smiles_extra_data' / 'JCIM_sup_bigsmiles.csv', usecols=['SMILES', 'Tg (C)'])
    train = add_extra_data(train, tg2_df, 'Tg')

    tg3_df = pd.read_excel(EXTERNAL / 'smiles_extra_data' / 'data_tg3.xlsx')
    tg3_df['Tg'] = tg3_df['Tg [K]'] - 273.15
    train = add_extra_data(train, tg3_df[['SMILES', 'Tg']], 'Tg')

    # Density
    dnst_df = pd.read_excel(EXTERNAL / 'smiles_extra_data' / 'data_dnst1.xlsx')
    train = add_extra_data(train, dnst_df[['SMILES', 'density(g/cm3)']], 'Density')

    # 3. Save to interim
    INTERIM.mkdir(parents=True, exist_ok=True)
    train.to_parquet(INTERIM / 'train.parquet', index=False)
    test.to_parquet(INTERIM / 'test.parquet', index=False)


if __name__ == '__main__':
    process_raw()
