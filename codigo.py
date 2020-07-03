import numpy as np
from matplotlib import pyplot as plt

# Funcion sigmoidea
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
        
# Lectura del archivo entrada de los datos
matriz=[]
with open('irisp.csv', 'r')  as archivo:
    lineas = archivo.read().splitlines()
    lineas.pop(0)
    for l in lineas:
        linea = l.split(';')      
        matriz.append([float(linea[0]),float(linea[1]),float(linea[2]),float(linea[3])])
datosEntrada = np.array(matriz)              

# Salida de la red        
e = []
r = []
for i in range(1):
    e.append(1)
for j in range(len(datosEntrada)):
    r.append(e)
datosSalida = np.array(r)

# Semilla random
np.random.seed(1)

# Numero de neuronas en la capa oculta
neuronasCO = 3

# Valor de la constante de aprendizaje
eta = 0.1

# Numero de iteraciones (Epoca) 
iteraciones = 100000

# Pesos aleatorios 
syn0 = 2*np.random.random((4,neuronasCO)) - 1
syn1 = 2*np.random.random((neuronasCO,3)) - 1
print('\nPesos en la primera capa:\n',syn0)
print('\nPesos en la segunda capa:\n',syn1)

# Vector con los datos de error absoluto para la grafica
error = []

# Vector con los datos de las iteraciones para la grafica
epoca = []

for iter in range(iteraciones):

    l0 = datosEntrada
    l1 = nonlin(np.dot(l0,syn0))
    l2 = nonlin(np.dot(l1,syn1))
    l2_error = datosSalida - l2
    a = np.mean(np.abs(l2_error))
    
    # Almacenamiento de los datos en los vectores para la grafica
    
    error.append(a)
    epoca.append(iter+1)

    # Entrenamiento
    
    if (iter% 10) == 0:
        print ('\nError Promedio Absoluto:\n' + str(np.mean(np.abs(l2_error))))

    l2_delta = l2_error * nonlin(l2,deriv=True)*eta
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * nonlin(l1,deriv=True)*eta
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
        
print ('\nSalida despues del entrenamiento:')
print('\nSalida de la red:\n',l2)
print ('\nError:' + str(np.mean(np.abs(l2_error))))

# Clasificacion de las tres clases de flor
j = 0        
for i in range(len(l2)):   
    if (l2[i][j]>l2[i][j+1] and l2[i][j]>l2[i][j+2]):
       l2[i][j]=1
       l2[i][j+1]=0
       l2[i][j+2]=0      
    elif (l2[i][j+1]>l2[i][j] and l2[i][j+1]>l2[i][j+2]):
       l2[i][j]=0
       l2[i][j+1]=1
       l2[i][j+2]=0      
    else:
       l2[i][j]=0
       l2[i][j+1]=0
       l2[i][j+2]=1

print ('\nPesos nuevos en la primera capa:\n',syn0)
print ('\nPesos nuevos en la segunda capa:\n',syn1)
            
# Tabla de las iteraciones  
print('\nTabla Epoca x Error:')    
for i in range(len(epoca)):
    print(epoca[i],' ----> ',error[i])

# Grafica Epoca x Error promedio    
plt.title('GRAFICA')
plt.plot(epoca, error, label = 'Error x Epoca')
plt.grid()
plt.legend()
plt.show()

# Impresion de la clasificacion 
print ('\n\nLA CLASIFICACION ES: \n')         
j = 0
for i in range(len(l2)):
    if (l2[i][j]==1):
        print(i+1,datosEntrada[i],'--->',l2[i],'---> Iris-Setosa')              
    elif (l2[i][j+1]==1):
        print(i+1,datosEntrada[i],'--->',l2[i],'---> Iris-Versicolor')       
    if(l2[i][j+2]==1):
        print(i+1,datosEntrada[i],'--->',l2[i],'---> Iris-Virginica')  




