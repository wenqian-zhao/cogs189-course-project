import pandas as pd
import numpy as np


def data_import():
    n_skip = 19
    url = "http://suendermann.com/corpus/EEG_Eyes.arff.gz"
    df = pd.read_csv(url, compression="gzip", header=None,skiprows=n_skip)

    attributes = [
        "AF3", "F7", "F3",
        "FC5", "T7", "P7",
        "O1", "O2", "P8",
        "T8", "FC6", "F4",
        "F8", "AF4"
    ]
    df.columns = attributes + ["label"]
    df.to_csv("data/data.csv", index=False)


