import pandas as pd
import os


def read_csv(filename):
    df = pd.read_csv(filename)
    return df


# if __name__ == "__main__":
#     path  = "../Dataset/Training/"
#     filepath = os.listdir(path)
#     dfs = []
#     dfs.append(read_csv(path + "Features_Variant_1.csv"))
#
#     # for file in filepath:
#     #     if file.endswith(".csv"):
#     #         dfs.append(read_csv(path + file))
#     print(";")
#     pass
