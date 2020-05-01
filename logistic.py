"""
En este archivo vemos un ejemplo de lo que es la Regreion Logistica 
aplicada en el machine learning, utilizando un archivo csv donde se encuentran
registros de usuarios de navegacion en una pagina web.
Con este script podemos predecir datos que obtuvimos previamente y entrenando un modelo
de regresion y saber que tipo de usuarios navegan nuestra website
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, model_selection
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)

dataframe = pd.read_csv("db.csv")
print(dataframe.groupby('clase').size())

dataframe.drop(['clase'],1).hist()
plt.show()

sb.pairplot(dataframe.dropna(), hue='clase',height=4,vars=["duracion", "paginas","acciones","valor"],kind='reg')
plt.show()

x = np.array(dataframe.drop(['clase'],1))
y = np.array(dataframe['clase'])
model = linear_model.LogisticRegression()
model.fit(x,y)

predictions = model.predict(x)
#print(predictions[0:5])

#Validacion de modelo utilizando 80% de registros y 20% de entrenamiento
validation_size = 0.20
seed = 7
x_train, x_validation, y_train, y_validation = model_selection.train_test_split(x,y,test_size=validation_size,random_state=seed)
name = 'Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=seed)
cv_results = model_selection.cross_val_score(model, x_train, y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
#print(msg)


#Prediccion y clasificacion de datos 
predictions = model.predict(x_validation)
#print(accuracy_score(y_validation, predictions))


"""
Los datos que se muestran dentro de la matriz de confusion deben de estar 
dentro de la diagonal para ser un dato valido, en dado caso que se encuentr
fuera de la diagonal, es un dato mal predecido.

"""
##Reporte de resultado del modelo
print(confusion_matrix(y_validation, predictions))

#print(classification_report(y_validation, predictions))


#Prueba de usuario ficticio
#x_new = pd.DataFrame({'duracion': [10], 'paginas': [3], 'acciones': [5], 'valor': [9]})
#print(model.predict(x_new))