import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# ==============================================================================
# 1. CARREGAR OS DADOS
# ==============================================================================
arquivo = 'dataset_FINAL_TREINO.csv'

try:
    df = pd.read_csv(arquivo)
    print(f"--> Dataset carregado: {len(df)} amostras.")
except FileNotFoundError:
    print(f"ERRO: Não encontrei '{arquivo}'. Verifique se rodou a fusão corretamente.")
    exit()

# ==============================================================================
# 2. PREPARAÇÃO (SEPARAR PROVA E ESTUDO)
# ==============================================================================
# X = Características (Potencia, Curtose, CN0_NMEA...)
X = df.drop('Classe', axis=1)
# y = Gabarito (0 = Autêntico, 1 = Spoofing)
y = df['Classe']

# Divide: 70% para a IA aprender (Treino), 30% para a prova final (Teste)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print(f"--> Treinando com {len(X_train)} amostras.")
print(f"--> Testando com {len(X_test)} amostras.")

# ==============================================================================
# 3. TREINAMENTO (O CÉREBRO DA OPERAÇÃO)
# ==============================================================================
print("--> Iniciando treinamento do Random Forest...")

# Cria o modelo com 100 árvores de decisão
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

print("--> Modelo treinado com sucesso!")

# ==============================================================================
# 4. AVALIAÇÃO E RESULTADOS
# ==============================================================================
# Faz a previsão nos dados de teste (que o modelo nunca viu)
y_pred = model.predict(X_test)

# Calcula a acurácia (Nota final)
acuracia = accuracy_score(y_test, y_pred)
print(f"\n{'='*40}")
print(f"ACURÁCIA FINAL: {acuracia*100:.2f}%")
print(f"{'='*40}\n")

# Mostra relatório detalhado (Precisão, Recall) no terminal
print("Relatório de Classificação:")
print(classification_report(y_test, y_pred, target_names=['Autêntico', 'Spoofing']))

# ==============================================================================
# 5. GRÁFICOS PARA O RELATÓRIO (TIRE PRINT DESTES!)
# ==============================================================================

# GRÁFICO A: Matriz de Confusão
# Mostra exatamente onde o modelo acertou e errou
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16},
            xticklabels=['Autêntico (Predito)', 'Spoofing (Predito)'], 
            yticklabels=['Autêntico (Real)', 'Spoofing (Real)'])
plt.title(f'Matriz de Confusão (Acurácia: {acuracia:.2%})', fontsize=14)
plt.tight_layout()
plt.show()

# GRÁFICO B: Importância das Características (Feature Importance)
# Esse gráfico prova que você usou o C/N0 e a Integridade do Sinal
importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
# Usa paleta de cores gradiente para destacar o mais importante
sns.barplot(x=importances.values, y=importances.index, hue=importances.index, legend=False, palette='viridis')
plt.title('Quais características foram decisivas?', fontsize=14)
plt.xlabel('Grau de Importância (0 a 1)')
plt.ylabel('Característica')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("\nConcluído! Use os gráficos acima na seção 'Resultados' do seu relatório.")

# ADICIONE ISTO NO FINAL:
import joblib

# Salva o modelo treinado em um arquivo .pkl (pickle)
nome_arquivo_modelo = 'meu_detector_gps.pkl'
joblib.dump(model, nome_arquivo_modelo)

print(f"\nMODELO SALVO! O arquivo '{nome_arquivo_modelo}' foi criado.")
print("Agora você pode usar esse arquivo para detectar spoofing em tempo real.")