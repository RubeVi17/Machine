import pandas as pd
import numpy as np
import matplotlib.pylab as plt

plt.rcParams['figure.figsize'] = (16,9)
plt.style.use('fast')

from keras.models import Sequential
from keras.layers import Dense,Activation,Flatten
from sklearn.preprocessing import MinMaxScaler

df = pd.read_csv('time_series.csv', parse_dates=[0], header=None, index_col=0, squeeze=True, names=['fecha', 'unidades'])

#print(df.describe())
"""
#Obtener promedio mensual de venta de unidades
meses = df.resample('M').mean()
#Graficamos las medidas
verano2017 = df['2017-06-01':'2017-09-01']
verano2018 = df['2018-06-01':'2018-09-01']
plt.plot(verano2017.values, color='blue', label='2017')
plt.plot(verano2018.values, color='red', label='2018')
plt.legend()
plt.show()
"""

"""
Creando red neuronal para ejcutar nuestro archivo
"""
pasos = 7
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    #input sequence
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1,i)) for j in range(n_vars)]

    #forecast sequence
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1,i)) for j in range(n_vars)]
    
    #put all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    #drop rows with nan value
    if dropnan:
        agg.dropna(inplace=True)
    return agg
"""
#load data sets
values = df.values
#ensure all data is float
values = values.astype('float32')
#normalize
scaler = MinMaxScaler(feature_range=(-1,1))
values=values.reshape(-1,1)
scaled = scaler.fit_transform(values)
#frame as supervised learning
reframed = series_to_supervised(scaled, pasos, 1)

#split into train and test
values= reframed.values
n_train_days = 315+289 -(30+pasos)
train = values[:n_train_days, :]
test = values[n_train_days:, :]
#split into input and outputs
x_train, y_train = train[:, :-1], train[:, -1]
x_val, y_val = test[:, :-1], test[:, -1]
#reshape input to be 3d
x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1]))
x_val = x_val.reshape((x_val.shape[0],1,x_val.shape[1]))
#print(x_train.shape, y_train.shape, x_val.shape, y_val.shape)
"""
def crear_modeloFF():
    model = Sequential()
    model.add(Dense(pasos, input_shape=(1,pasos),activation='tanh'))
    model.add(Flatten())
    model.add(Dense(1,activation='tanh'))
    model.compile(loss='mean_absolute_error', optimizer='Adam', metrics=["mse"])
    model.summary()
    return model


#Entrenamiento de modelo
epochs=40
model = crear_modeloFF()
#history = model.fit(x_train,y_train,epochs=epochs,validation_data=(x_val,y_val),batch_size=pasos)
"""
results = model.predict(x_val)
plt.scatter(range(len(y_val)),y_val,c='g')
plt.scatter(range(len(results)),results,c='r')
plt.title('Validate')
plt.show()
"""

"""
Crear pronostico de ventas
"""

ultimosDias = df['2018-11-16':'2018-11-30']

values = ultimosDias.values
values = values.astype('float32')
#normalize features
values = values.reshape(-1,1)
scaler = MinMaxScaler(feature_range=(-1,1))
scaled = scaler.fit_transform(values)
reframed = series_to_supervised(scaled, pasos, 1)
reframed.drop(reframed.columns[[7]], axis=1, inplace=True)

values = reframed.values
x_test = values[6:, :]
x_test = x_test.reshape((x_test.shape[0],1,x_test.shape[1]))


def aggValue(x_test, nVal):
    for i in range(x_test.shape[2]-1):
        x_test[0][0][i]=x_test[0][0][i+1]
    x_test[0][0][x_test.shape[2]-1]=nVal
    return x_test

results=[]
for i in range(7):
    parcial = model.predict(x_test)
    results.append(parcial[0])
    x_test=aggValue(x_test, parcial[0])

adimen = [x for x in results]
inverted = scaler.inverse_transform(adimen)

prediccion1SemanaDic = pd.DataFrame(inverted)
prediccion1SemanaDic.columns = ['Pronostico']
prediccion1SemanaDic.plot()
diciembre2017 = df['2017-12-01':'2017-12-08']
plt.plot(diciembre2017.values, color='red', label='2017')
plt.legend()
plt.show()

