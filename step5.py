import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

time_encoding = pd.read_csv("step3/time_encoding.csv", sep=",", low_memory=False)

y = time_encoding['grav']

features = ['catu','sexe','trajet','secu',
            'catv','an_nais','mois',
            'occutc','obs','obsm','choc','manv',
            'lum','agg','int','atm','col','gps',
            'catr','circ','vosp','prof','plan',
            'surf','infra','situ','hrmn','geo']

X_train_data = pd.get_dummies(time_encoding[features].astype(str))

X_train_data.to_csv("step5/one_hot_encoding.csv", index=False)