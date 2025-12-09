import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# --- 1. CARREGAR DADOS ---
ARQUIVO_CSV = 'Spoofing-Detection-using-GNSS-Signals/Versão2/features_normalizadas_com_fft3.csv'

try:
    df = pd.read_csv(ARQUIVO_CSV)
    print(f"Dados carregados! Total de amostras: {len(df)}")
except FileNotFoundError:
    print("ERRO: O arquivo CSV não foi encontrado. Rode o script de extração de features primeiro.")
    exit()

# --- 2. PREPARAÇÃO (SPLIT) ---
# Separa as features (X) do alvo (y)
X = df.drop('Classe', axis=1)
y = df['Classe']

# Divide em Treino (70%) e Teste (30%)
# random_state garante que o resultado seja reproduzível
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# --- 3. TREINAMENTO (RANDOM FOREST) ---
print("Treinando Random Forest...")
# n_estimators=100 cria 100 árvores de decisão
clf = RandomForestClassifier(n_estimators=800, random_state=42)
clf.fit(X_train, y_train)

# --- 4. AVALIAÇÃO ---
y_pred = clf.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"\n--- RESULTADOS ---")
print(f"Acurácia: {acuracia:.2%}")
print("\nRelatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Autêntico', 'Spoofing']))

# --- 5. VISUALIZAÇÃO ---
plt.figure(figsize=(14, 6))

# Gráfico 1: Matriz de Confusão
plt.subplot(1, 2, 1)
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Autêntico', 'Spoofing'], 
            yticklabels=['Autêntico', 'Spoofing'])
plt.title('Matriz de Confusão')
plt.ylabel('Verdadeiro')
plt.xlabel('Predito')

# Gráfico 2: Importância das Features (O MAIS IMPORTANTE)
plt.subplot(1, 2, 2)
importancias = clf.feature_importances_
nomes_features = X.columns
indices = np.argsort(importancias)[::-1] # Ordena do maior para o menor

sns.barplot(x=importancias[indices], y=nomes_features[indices], palette="viridis")
plt.title('Importância das Features (O que o modelo "olhou"?)')
plt.xlabel('Importância Relativa')

plt.tight_layout()
plt.show()