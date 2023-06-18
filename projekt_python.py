import random

class Data:
    def __init__(self, distance, restaurant_rating, delivery_time, additional_info=None):
        self.distance = distance
        self.restaurant_rating = restaurant_rating
        self.delivery_time = delivery_time
        self.additional_info = additional_info

    def __str__(self):
        return f"Data: distance={self.distance}, restaurant_rating={self.restaurant_rating}, delivery_time={self.delivery_time}"

class DataLoader:
    def load_data(self):
        data = [
            (2.3, 4.5, 23.5, {'customer_name': 'John Doe', 'order_date': '2023-06-10', 'order_id': '12345'}),
            (1.8, 3.9, 18.2, {'customer_name': 'Jane Smith', 'order_date': '2023-06-11', 'order_id': '67890'}),
            (3.1, 4.2, 29.1, {'customer_name': 'Alice Johnson', 'order_date': '2023-06-12', 'order_id': '54321'})
        ]
        return [Data(*record) for record in data]

def calculate_coefficients(X, y):
    n = len(X)
    sum_x = sum_y = sum_xy = sum_xx = 0

    for i in range(n):
        sum_x += X[i]
        sum_y += y[i]
        sum_xy += X[i] * y[i]
        sum_xx += X[i] * X[i]

    slope = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x)
    intercept = (sum_y - slope * sum_x) / n

    return slope, intercept

def predict_delivery_time(distance, restaurant_rating, slope, intercept):
    return slope * distance + intercept

def calculate_mean_squared_error(y_true, y_pred):
    n = len(y_true)
    mse = sum((y_true[i] - y_pred[i])**2 for i in range(n)) / n
    return mse

def measure_execution_time(func):
    import time

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"Execution time of {func.__name__}: {execution_time:.4f} seconds")
        return result

    return wrapper

@measure_execution_time
def main():
    # Tworzenie instancji obiektu DataLoader i wczytywanie danych
    data_loader = DataLoader()
    data = data_loader.load_data()

    # Przygotowanie danych do trenowania modelu
    X = [item.distance for item in data]
    y = [item.delivery_time for item in data]

    # Obliczanie współczynników regresji liniowej
    slope, intercept = calculate_coefficients(X, y)

    # Przykłady danych dostawy
    examples_data = [
        (5.4, 4.3, None),  # Przykład 1
        (2.5, 3.8, None),  # Przykład 2
        (3.8, 4.5, None),  # Przykład 3
        (9.0, 4.1, None)   # Przykład 4
    ]

    for example in examples_data:
        distance, restaurant_rating, target = example
        predicted_delivery_time = predict_delivery_time(distance, restaurant_rating, slope, intercept)
        print('Predicted Delivery Time:', predicted_delivery_time)

    # Ocena modelu
    y_pred = [predict_delivery_time(item.distance, item.restaurant_rating, slope, intercept) for item in data]
    mse = calculate_mean_squared_error(y, y_pred)
    print('Mean Squared Error:', mse)

if __name__ == '__main__':
    main()

# Losowe wyświetlanie klienta
data_loader = DataLoader()
random_data = random.choice(data_loader.load_data())
random_customer_name = random_data.additional_info['customer_name']
print(random_customer_name)

#Koniec kodu