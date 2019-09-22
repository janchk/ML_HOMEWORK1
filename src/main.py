import numpy as np
import pandas as pd
from src.parse_csv import read_csv

if __name__ == "__main__":
    path = "../Dataset/Training/"
    csv_file = path + "Features_Variant_1.csv"
    df = read_csv(csv_file)
    df_array = df.values
    # X = [row[0:28, 35:53] for row in df_array]
    X = [np.hstack((row[0:29], row[34:54])) for row in df_array]
    Y = [row[29] for row in df_array]



