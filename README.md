# inteligencia artificial Gustavo
Nome: Henrique santos de jesus 2-C

import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 


boston_csv = "https://raw.githubusercontent.com/selva86/datasets/ma"
boston = pd.read_csv(boston_csv)
#print(boston.head())

#Criar os vetores de recursos e alvo
x = boston.drop('medv', axis=1).valves
y = boston['medv'].valves

# prevendo o preço a partir de um único recurso

x_rooms = x[:, 5]
print(x)
print(type(x_rooms))
print(y)
print(type(y))

x_rooms = x_rooms.reshape(1-, 1)
y = y.reshape(-1, 1)

print(x)
print(y)

# Valor médio vs n de quarto
plt,scatter(x_rooms, y) 
plt.ylabel('Valor da casa/1000 ($)')
plt.slabel('Número de quartos')
plt.show()

reg = linear_model.linearRegression() 
reg.fit(x_rooms, y)
prediction_space = np.linspace(min,(x_rooms), max(x_rooms)).reshape(-1, 1)

plt.scatter(x_rooms, y, color='blue')
plt.plot(prediction_space, reg.predict(prediction_space), color='black', linewleth=)
plt.show()

from sklearn import datasets 
import matplotlib.pyplot as plt 
from sklearn.neighbors import KNeighborsClassifler 
from sklearn.model_selection import train_test_split 

plt.style.use('ggplot')

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train , y_train, x_test, y_test, = train_test_split(x, y, test_size=0,3, random_state=7, stratify=y)


Knn = KNeighborsClassifler(n_neighbors=6)

Knn,fit(x_train, y_train)

print(Knn,score(x_test, y-test))
y_pred = knn_predict(y_test) 
x_new = [[7.2,5.1,2.4,]]
 from sklearn import datasets
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifler 
from sklearn.model_selection import train_test_split 

plt.style.use('ggplot')

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train , y_train, x_test, y_test, = train_test_split(x, y, test_size=0,3, random_state=7, stratify=y)


Knn = KNeighborsClassifler(n_neighbors=6)

Knn,fit(x_train, y_train)

print(Knn,score(x_test, y-test))
y_pred = knn_predict(y_test) 
x_new = [[7.2,5.1,2.4,]]

   Class Lampada: 
  def _init_(self):
        self.estado = False  # Começa desligada

    def alterar _estado(self):
      if self.estado == False:
          self.estado = True
     else:
         self.estado = False

# Uso da Classe
 lampada = Lampada()
 print(Lampada.estado) # False
 print(Lampada.estado) # False
 print(Lampada.alterar_estado()) # Liga a lâmpada
 print(Lampada.estado)
 print(Lampada.alterar_estado()) # Desliga a lâmpada 
 print(Lampada.estado)

def caixa_eletronico(valor):
    notas = [50, 20, 10, 5]
    quantidade_notas = [0, 0, 0, 0]
    restante = valor
    
    for i in range(len(notas)):
        while restante >= notas[i]:
            quantidade_notas[i] += 1
            restante -= notas[i]
    
    print("Quantidade de notas:")
    for i in range(len(notas)):
        if quantidade_notas[i] > 0:
            print(f"{quantidade_notas[i]} nota(s) de R${notas[i]}")

if __name__ == "__main__":
    valor = float(input("Digite o valor que deseja sacar: "))
    if valor <= 0:
        print("Valor inválido. Por favor, insira um valor positivo.")
    else:
        caixa_eletronico(valor)

        numero = 1
        while True:
          print(numero)
          numero + = 1
         if (numero) > 10:
