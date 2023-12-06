import pandas as pd
import matplotlib.pyplot as plt

# Votre code existant pour lire et traiter les données
missing_values_deleted = pd.read_csv("step1/merged_data.csv", sep=",", low_memory=False)
nans = ['v1', 'v2', 'lartpc', 'larrout', 'locp', 'etatp', 'actp', 'voie', 'pr1', 'pr', 'place']
missing_values_deleted = missing_values_deleted.drop(columns=nans)
missing_values_deleted = missing_values_deleted.dropna()

# Assurez-vous que la colonne que vous souhaitez visualiser existe dans le DataFrame
# Remplacez 'nom_de_la_colonne' par le nom réel de la colonne que vous souhaitez visualiser
colonne_a_visualiser = 'Num_Acc'

# Vérifiez si la colonne existe dans le DataFrame
if colonne_a_visualiser in missing_values_deleted.columns:
    # Créez votre graphique (exemple avec un histogramme)
    plt.hist(missing_values_deleted[colonne_a_visualiser], bins=20)
    plt.title('Titre de votre graphique')
    plt.xlabel('Axe X')
    plt.ylabel('Axe Y')

    # Sauvegardez le graphique au format .png dans le dossier 'step2'
    plt.savefig('step2/mon_graphique.png')
    missing_values_deleted.to_csv("step2/missing_values_deleted.csv", index=False)
else:
    print(f"La colonne '{colonne_a_visualiser}' n'existe pas dans le DataFrame.")