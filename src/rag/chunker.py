import pandas as pd

def sort_chunk(dataset_path):
    # 1. On charge le dataset à l'intérieur de la fonction.
    dataset = pd.read_csv(dataset_path, sep=",") 

    # 2. On prépare les boîtes vides
    L_SF7 = []
    L_SF8 = []
    L_SF9 = []
    L_SF10 = []
    L_SF11 = []
    L_SF12 = []
    L_DR8 = []
    L_DR9 = []

    # 3. La boucle iterrows pour lire ligne par ligne
    for _, row in dataset.iterrows():
        # On extrait la valeur de la colonne "modulation" pour cette ligne
        mod = row["modulation"] 

        # 4. Les conditions elif
        if mod == "SF7":
            L_SF7.append(row)
        elif mod == "SF8":
            L_SF8.append(row)
        elif mod == "SF9":
            L_SF9.append(row)
        elif mod == "SF10":
            L_SF10.append(row)
        elif mod == "SF11":
            L_SF11.append(row)
        elif mod == "SF12":
            L_SF12.append(row)
        elif mod == "DR8":
            L_DR8.append(row)
        elif mod == "DR9":
            L_DR9.append(row)

    # 5. On renvoie les listes remplies
    return L_SF7, L_SF8, L_SF9, L_SF10, L_SF11, L_SF12, L_DR8, L_DR9

# --- Pour lancer la fonction : ---
def init(main):
    SF7_data, SF8_data, SF9_data, SF10_data, SF11_data, SF12_data, DR8_data, DR9_data = sort_chunk()
    print(f"rangé {len(SF7_data)} lignes dans L_SF7.")

def chunker_by_physic_similarity(dataset_path):
    pass

