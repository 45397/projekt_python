#Projekt tworzony w Visual Code Studio
#Pakiety potrzebne do działania programu poprawnie: Pandas oraz Sklearn
#Do zainstalowania pandas: pip install pandas w CMD
#Do zainstalowania sklearn: pip install scikit-learn CMD

#Odpowiada za wczytywanie, przetwarzanie danych
import pandas as pd
import csv

#Implementacja algorytmów Machine Learning
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

#Wczytywanie danych z tabeli "delivery_data.csv", dane wygenerowane randomowo
data = pd.read_csv('delivery_data.csv')

#Podział danych na cechy i docelowy "czas dostawy jedzenia"
x = data[['distance', 'restaurant_rating']] 
y = data['delivery_time']  

#Testowanie, 40%procent danych z naszych 100 w tabeli, zostanie użyte jako testowanie modelu, pozostałe będzie wykorzastane jako trening
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=50)

#Model
model = LinearRegression()

#Trenowanie modelu
model.fit(x_train, y_train)

#Predykcja
y_pred = model.predict(x_test)

#Wynik testu programu
mse = mean_squared_error(y_test, y_pred)
print('Przybliżony czas dostawy:', mse)
