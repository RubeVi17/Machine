#Este solo es un archivo para aprender machine learning
from matplotlib import pyplot as plt

x1 = [3,4,5,6]
y1 = [5,6,3,4]
x2 = [2,5,8]
y2 = [3,4,3]

plt.plot(x1,y1, color='blue', linewidth=4, label='Linea 1')
plt.plot(x2,y2, color='green', linewidth=4, label='Linea 2')

plt.title('Diagrama de lineas')
plt.ylabel('Eje Y')
plt.xlabel('Eje X')

plt.legend()
plt.grid()
plt.show()


x1 = [0.25,1.25,2.25,3.25,4.25]
y1 = [10,55,80,32,40]
x2 = [0.75,1.75,2.75,3.75,4.75]
y2 = [42,26,10,29,66]

plt.scatter(x1,y1, label='Datos 1', color='red')
plt.scatter(x2,y2, label='Datos 2', color ='blue')
plt.title('Grafico de dispersion')
plt.legend()
plt.show()