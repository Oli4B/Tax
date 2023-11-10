import torch
import pandas as pd
from torch.utils.data import TensorDataset

DATA_FILE = "abt_ssl.feather" # File containing all data
DROP_COLS = [
    ""
]
DUMMY_COLS = [
]
CAT_COLS = [
]
ALL_COLS = None
CATS = None


def create_dataset(data, target_column="correctie_pos_ind"):
    """ Retrieve the data and target columns for specified sets """
    data = data.sample(frac=1, random_state=42)

    data = data.drop(DROP_COLS, axis=1)

    target = data[target_column]
    target = torch.tensor(target.values)

    data = data.drop(target_column, axis=1)
    input = torch.tensor(data.values).float()

    return input, target


def load_data(unlabelled=False):
    df = pd.read_feather(DATA_FILE)
    df = pd.get_dummies(df, columns=DUMMY_COLS, dummy_na=True)
    df = df.astype(float)

    global ALL_COLS
    ALL_COLS = [col for col in df.columns if col not in DROP_COLS]

    global CAT_COLS
    for colname in DUMMY_COLS:
        dummy_cols = [d for d in df.columns if colname in d]
        CAT_COLS.extend(dummy_cols)

    global CATS
    CATS = [int(df[col].max()+1) for col in CAT_COLS]

    # Normalize the data, but skip columns we will not use.
    for column in df.columns:
        if column in DROP_COLS:
            continue
        c = df[column]
        df[column] = (c-c.min())/(c.max()-c.min())


    if unlabelled:
        unlabelled_data = df.loc[
            (df["belastingjaar"] == 2019) & 
            (df["ind_stkprf"] == 0) & 
            (df["bedrag_correctie"].isna())]

    labelled_data_stk = df.loc[
        (df["belastingjaar"] == 2019) & 
        (df["ind_stkprf"] == 1) | 
        (df["belastingjaar"] == 2020) | 
        (df["belastingjaar"] == 2021)]
    
    labelled_data_toets = df.loc[
        (df["belastingjaar"] == 2019) & 
        (~df["bedrag_correctie"].isna())]

    del df

    if unlabelled:
        unlabelled_data = create_dataset(unlabelled_data)[0]
    labelled_data_stk = TensorDataset(*create_dataset(labelled_data_stk))
    labelled_data_toets = TensorDataset(*create_dataset(labelled_data_toets))

    print("Data loaded")
    print("labelled stk data:  ", len(labelled_data_stk),   " items")
    print("labelled toe data:  ", len(labelled_data_toets), " items")
    
    if unlabelled:
        return unlabelled_data, labelled_data_stk, labelled_data_toets
    return labelled_data_stk, labelled_data_toets
