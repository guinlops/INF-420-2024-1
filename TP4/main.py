import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

try:
   df = pd.read_csv('koi_data.csv')
except:
   print("Falha ao abrir o banco de dados!")
else:
   print("Banco de dados aberto!")

#Tipo de dado de cada coluna
df.info()


#Contagem de quantos CONFIRMED E FALSE POSITIVE tem NO DF

contagens = df.koi_disposition.value_counts()

# Criando a figura e o eixo
fig, ax = plt.subplots()

# Ocultando o eixo (opcional)
ax.axis('off')
ax.axis('tight')

# Convertendo para DataFrame para criar a tabela
table_data = contagens.reset_index()
table_data.columns = ['koi_disposition', 'Contagem']

# Criando a tabela
ax.table(cellText=table_data.values, colLabels=table_data.columns, cellLoc='center', loc='center')

# Ajustando o layout
fig.tight_layout()

# Exibindo a tabela
plt.show()

X = df.iloc[:, 2:].values  # Todas as colunas exceto a primeira (KOI ID) e a segunda (label)
Y = df.iloc[:, 1].values   # Segunda coluna (label) Koi Disposition, Confirmed, False Positive

#Usando KFold para Validação Cruzada com k igual a 5.


models = {
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(),
    'k-Nearest Neighbors': KNeighborsClassifier(),
    'Support Vector Machines': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Gradient Tree Boosting': GradientBoostingClassifier(),
    #'Multi-layer Perceptron': MLPClassifier()  # Aumentar o número de iterações
    'Multi-layer Perceptron': MLPClassifier(
        hidden_layer_sizes=(25,),
        activation='tanh',
        solver='adam',
        learning_rate_init=0.003,
        max_iter=1000
    )
}

results = {}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

for model_name, model in models.items():
    # Cria o pipeline com a normalização e o modelo
    pipeline = make_pipeline(StandardScaler(), model)
    
    # Executa a validação cruzada k-fold
    scores = cross_val_score(pipeline, X, Y, cv=kf, scoring='accuracy')
    
    # Exibe a acurácia média e o desvio padrão
    print(f'{model_name}: Acurácia média = {scores.mean():.4f}, Desvio padrão = {scores.std():.4f}')