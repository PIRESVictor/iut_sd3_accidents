import pandas as pd


carac = pd.read_csv("data/carac.csv", sep=";")
lieux = pd.read_csv("data/lieux.csv", sep=";", low_memory=False)
veh = pd.read_csv("data/veh.csv", sep=";")
vict = pd.read_csv("data/vict.csv", sep=";")


victime = vict.merge(veh, on=["Num_Acc", "num_veh"])
accident = carac.merge(lieux, on="Num_Acc")
victime = victime.merge(accident, on="Num_Acc")

# Merger les dataframes dans un dataframe merged_data.csv
victime.to_csv("step1/merged_data.csv", index=False)