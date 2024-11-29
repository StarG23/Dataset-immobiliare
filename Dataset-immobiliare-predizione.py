import numpy as np
import pandas as pd
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


# -- CREAZIONE DEL DATASET IMMOBILIARE --

# Generazione di un dataset sintetico con 2000 campioni
n_samples = 2000
superficie = np.random.randint(29, 200, n_samples)  # Superficie in m²
stanze = np.random.randint(1, 4, n_samples)  # Numero di stanze
età_fabbricato = np.random.randint(0, 100, n_samples)  # Età del fabbricato
distanza_centro = np.random.randint(1, 15, n_samples)  # Distanza in km dal centro

# Prezzo sintetico calcolato con una funzione lineare e rumore ridotto
prezzo = (
    100000 
    + superficie * 1000
    + stanze * 5000
    - età_fabbricato * 500
    + distanza_centro * 1000
    + np.random.normal(0, 20000, n_samples)  # Rumore ridotto
)

# Creazione del DataFrame
data = pd.DataFrame({
    'Superficie': superficie,
    'Numero di stanze': stanze,
    'Età fabbricato': età_fabbricato,
    'Distanza dal centro': distanza_centro,
    'Prezzo': prezzo
})

# Arrotondamento della colonna Prezzo a 2 decimali
data['Prezzo'] = data['Prezzo'].round(2)

# Visualizza il dataset
print(data)

# Rimozione degli outlier
limite_superiore = data['Prezzo'].quantile(0.99)
data = data[data['Prezzo'] <= limite_superiore]

# Salvataggio del dataset su file CSV
data.to_csv('dataset_case.csv', index=False)

# Visualizzazione della distribuzione dei prezzi
plt.figure(figsize=(10, 6))
sns.histplot(data['Prezzo'], kde=True, bins=30, color='skyblue')
plt.title("Distribuzione dei Prezzi degli Immobili")
plt.xlabel("Prezzo (€)")
plt.ylabel("Frequenza")
plt.show()


# -- PREPARAZIONE DEI DATI --

# Separazione di variabili indipendenti (X) e dipendenti (Y)
X = data[['Superficie', 'Numero di stanze', 'Distanza dal centro', 'Età fabbricato']].values
Y = data[['Prezzo']].values

# Suddivisione in training e test set
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Normalizzazione dei dati
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# -- CREAZIONE DEL MODELLO RETE NEURALE --

# Definizione del modello di rete neurale
model = Sequential([
    Dense(128, activation='relu', input_dim=4),  # Input layer
    Dense(64, activation='relu'),  # Hidden layer
    Dense(32, activation='relu'),  # Hidden layer
    Dense(1)  # Output layer
])

# -- ADDESTRAMENTO E VALUTAZIONE MODELLO --

# Compilazione del modello con una funzione di perdita meno sensibile agli outlier
model.compile(optimizer='adam', loss='mean_absolute_error')

# Addestramento del modello
model.fit(X_train, Y_train, epochs=500, batch_size=64, validation_data=(X_test, Y_test), verbose=1)

# Valutazione sul test set
loss = model.evaluate(X_test, Y_test, verbose=0)
print(f"Mean Absolute Error sul Test Set: {loss:.2f}€")


# -- PREDIZIONE CON INPUT UTENTE --

# Funzione per fare predizioni basate su input utente
def input_utente():
    superficie = float(input("Inserisci la superficie in mq: "))
    stanze = int(input("Inserisci il numero di stanze: "))
    età_fabbricato = int(input("Inserisci l'età del fabbricato: "))
    distanza_centro = float(input("Inserisci la distanza dal centro in km: "))
    return np.array([[superficie, stanze, distanza_centro, età_fabbricato]])

# Predizione per un nuovo immobile
nuovo = input_utente()
nuovo_normalizzato = scaler.transform(nuovo)
predizione = model.predict(nuovo_normalizzato)
print(f"Prezzo predetto: {predizione[0][0]:.2f}€")


# -- ANALISI: PREDIZIONE VS DATASET --

plt.figure(figsize=(10, 6))
sns.histplot(data['Prezzo'], kde=True, bins=30, color='green', label='Prezzi dataset')
plt.axvline(predizione[0][0], color='red', linestyle='--', linewidth=2, label=f"Prezzo Predetto: {predizione[0][0]:.2f}€")
plt.title("Distribuzione Prezzi dataset vs Prezzo Predetto")
plt.xlabel("Prezzo (€)")
plt.ylabel("Frequenza")
plt.legend(fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()


# -- ANALISI: CONFRONTO CON MEDIA DEL DATASET --

media_prezzo = data['Prezzo'].mean()
scarto = predizione[0][0] - media_prezzo

if scarto > 0:
    print(f"Il prezzo predetto è superiore alla media del dataset di {scarto:.2f}€ (Media: {media_prezzo:.2f}€).")
elif scarto < 0:
    print(f"Il prezzo predetto è inferiore alla media del dataset di {abs(scarto):.2f}€ (Media: {media_prezzo:.2f}€).")
else:
    print(f"Il prezzo predetto è esattamente uguale alla media del dataset ({media_prezzo:.2f}€).")
