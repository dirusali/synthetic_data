#Tenemos un problema de clasificación supervisado (datos de training con clase 1=real y 0=sintético)
#Tras hacer plots de los datos hago una prueba de divergencia de Jensen-Shanon de las desviaciones típicias que me da 0.02
#Quiere decir que los datos son extremadamente parecidos, métodos como regresión logística y SVM consigo distinguir satisfactoriamente
#Como los datos son muy muy similares creo que lo mejor es usar tensforflow y keras para buscar patrones ocultos

#Importo las librerías que vamos a necesitar
import numpy as np
import csv
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import RMSprop

#Utilizo el dataset training para entrenar el modelo con pandas datafrae, el target es la columna class, X el resto
df = pd.read_csv('train.csv')
y = df['class']
X = df.drop('class',axis=1)

#Construyo y compilo mi modelo de red neuronal con Keras ,Uso entropía binaria xq las clases son 0 y 1
#Pruebo varios graphs hasta obtener buena precisión y me quedo con 500 nodos layer 1, input 260 variables, 1 nodo en output

model = Sequential()
model.add(Dense(500, input_dim=260, activation='relu'))
model.add(Dense(260, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x=X,y=y,batch_size=100, epochs=10,shuffle=True)

#Hago un print de la precisión obtenida y me da 98.83 lo cual es buen resultado
_, accuracy = model.evaluate(X, y)
print('Accuracy: %.2f' % (accuracy*100))

#Comprobé un print de las predicciones del training y efectivamente coincidían con las del training
#Así que importo el test dataset para hacer las predicciones
df_test = pd.read_csv('test.csv')
X = df_test
predictions = model.predict(X)

#guardo los resultados en una lista 
resultados = []
for i in predictions:
    resultados.append(i[0])
  
#exporto los resultados a un csv  
with open('/home/vant/Escritorio/DiegoRuiz.csv', 'w', ) as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for word in resultados:
        wr.writerow([word])
    
