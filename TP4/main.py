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
#     'Multi-layer Perceptron': MLPClassifier(
#         hidden_layer_sizes=(25,),
#         activation='tanh',
#         solver='adam',
#         learning_rate_init=0.003,
#         max_iter=1000
#     )
}



def initial_Count():
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



def acuracia_media():
    results = {}
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, model in models.items():
        # Cria o pipeline com a normalização e o modelo
        pipeline = make_pipeline(StandardScaler(), model)
        
        # Executa a validação cruzada k-fold
        scores = cross_val_score(pipeline, X, Y, cv=kf, scoring='accuracy')
        
        # Armazena a acurácia média
        print(f'{model_name}: Acurácia média = {scores.mean():.4f}, Desvio padrão = {scores.std():.4f}')
        results[model_name] = scores.mean()

    # Criando o gráfico de barras para comparar as acurácias médias
    fig, ax = plt.subplots()
    model_names = list(results.keys())
    mean_accuracies = list(results.values())

    bars = ax.barh(model_names, mean_accuracies, color='red')

    # Adicionando marcações mais precisas no eixo x
    ax.set_xlabel('Acurácia Média')
    ax.set_title('Comparação das Acurácias Médias dos Modelos')

    # Ajustando as marcações do eixo x
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.4f}'))

    # Adicionando anotações para os valores máximos
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                f'{width:.4f}', va='center', ha='left')

    plt.show()



def naive_bayes_experiment(X, Y, kf):
    model = GaussianNB()
    pipeline = make_pipeline(StandardScaler(), model)
    scores = cross_val_score(pipeline, X, Y, cv=kf, scoring='accuracy')
    mean_accuracy = scores.mean()
    print(f'Naive Bayes: Acurácia média = {mean_accuracy:.4f}, Desvio padrão = {scores.std():.4f}')
    return mean_accuracy



def decision_tree_experiment(X, Y, kf):
    depths = [None, 5, 10, 15, 20]
    accuracies = []

    for depth in depths:
        model = DecisionTreeClassifier(max_depth=depth, random_state=42)
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipeline, X, Y, cv=kf, scoring='accuracy')
        accuracies.append(scores.mean())

    plt.plot(depths, accuracies, marker='o')
    plt.xlabel('Profundidade Máxima')
    plt.ylabel('Acurácia Média')
    plt.title('Impacto da Profundidade Máxima na Acurácia do Decision Tree')
    plt.grid(True)
    plt.show()

def svm_experiment(X, Y, kf):
    kernels = ['linear', 'rbf']
    accuracies = []

    for kernel in kernels:
        model = SVC(kernel=kernel, random_state=42)
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipeline, X, Y, cv=kf, scoring='accuracy')
        accuracies.append(scores.mean())

    plt.bar(kernels, accuracies, color=['blue', 'green'])
    plt.xlabel('Kernel')
    plt.ylabel('Acurácia Média')
    plt.title('Comparação entre Kernels do SVM')
    plt.grid(True)
    plt.show()



def knn_experiment(X, Y, kf):
    k_values = [1, 3, 5, 7, 9, 11]
    accuracies = []

    for k in k_values:
        model = KNeighborsClassifier(n_neighbors=k)
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipeline, X, Y, cv=kf, scoring='accuracy')
        accuracies.append(scores.mean())

    plt.plot(k_values, accuracies, marker='o')
    plt.xlabel('Número de Vizinhos (k)')
    plt.ylabel('Acurácia Média')
    plt.title('Impacto do Número de Vizinhos na Acurácia do k-NN')
    plt.grid(True)
    plt.show()



def random_forest_experiment(X, Y, kf):
    n_trees = [10, 50, 100, 200, 300]
    accuracies = []

    for n in n_trees:
        model = RandomForestClassifier(n_estimators=n, random_state=42)
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipeline, X, Y, cv=kf, scoring='accuracy')
        accuracies.append(scores.mean())

    plt.plot(n_trees, accuracies, marker='o')
    plt.xlabel('Número de Árvores')
    plt.ylabel('Acurácia Média')
    plt.title('Impacto do Número de Árvores na Acurácia do Random Forest')
    plt.grid(True)
    plt.show()

def mlp_experiment(X, Y, kf):
    activations = ['identity', 'logistic', 'tanh', 'relu']
    accuracies = []

    for activation in activations:
        model = MLPClassifier(
            hidden_layer_sizes=(25,),
            activation=activation,
            solver='adam',
            learning_rate_init=0.003,
            max_iter=1000,
            random_state=42
        )
        pipeline = make_pipeline(StandardScaler(), model)
        scores = cross_val_score(pipeline, X, Y, cv=kf, scoring='accuracy')
        accuracies.append(scores.mean())

    plt.bar(activations, accuracies, color='purple')
    plt.xlabel('Função de Ativação')
    plt.ylabel('Acurácia Média')
    plt.title('Comparação entre Funções de Ativação do MLP')
    plt.grid(True)
    plt.show()

def main():
      initial_Count()  # Certifique-se de que essa função está definida
      acuracia_media()

      kf = KFold(n_splits=5, shuffle=True, random_state=42)
#Aqui, shuffle=True embaralha os dados antes de dividí-los, e random_state=42 garante que os resultados sejam reprodutíveis.

      naive_bayes_experiment(X, Y, kf)
      svm_experiment(X, Y, kf)
    
      decision_tree_experiment(X, Y, kf)

      knn_experiment(X, Y, kf)
      random_forest_experiment(X, Y, kf)
      #mlp_experiment(X, Y, kf)
    
      return 0

if __name__ == '__main__':
    main()