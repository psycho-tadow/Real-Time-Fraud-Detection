from pathlib import Path
import pandas as pd

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "creditcard.csv"

def load_data(path=DATA_PATH,drop_duplicates=True):
    df=pd.read_csv(path)
    if drop_duplicates:
        df=df.drop_duplicates()
    return df

def time_split(df,train_frac=0.70,valid_frac=0.15):
    df=df.sort_values("Time").reset_index(drop=True)
    i1=int(len(df)*train_frac)
    i2=int(len(df)*(train_frac+valid_frac))
    return df.iloc[:i1],df.iloc[i1:i2],df.iloc[i2:]


def get_xy(df):
    X=df.drop(columns=["Class","Time"])
    y=df["Class"]
    return X,y
