# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 13:23:15 2017

@author: Laura
"""
import pandas as pd
from matplotlib import pyplot as plt

datos = pd.read_csv('wine.csv',header = 0)
print (datos)

vinoCabecera = ['Clase 1','Clase 2','Clase 3']

x = datos.ix[:,0]
y = datos.ix[:,1]

print (x)
print (y)

lista1 = x
lista2 = y

plt.plot(lista1,lista2,'green')
plt.title('Clasificacion del Vino Malo - Bueno - Regular')
plt.ylabel('Alcohol')
plt.xlabel('Clase de vino')
plt.show