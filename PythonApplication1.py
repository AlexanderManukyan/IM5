import pandas as pd 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import scipy.stats as stats

# Загрузка данных
def load_data(file_path):
    data = pd.read_csv(file_path)
    subset_data = data[['Confirmed', 'Recovered', 'Active', 'Lat', 'Long', 'Deaths']].dropna()
    return subset_data

# Проверка корреляций
def check_correlations(data):
    correlation_matrix = data.corr()
    print("\nМатрица корреляций:")
    print(correlation_matrix)

# Обучение линейной модели
def train_model(data):
    X = data[['Confirmed', 'Recovered', 'Active', 'Lat', 'Long']]
    y = data['Deaths']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)

    print("\nОценка модели:")
    print(f"Среднеквадратичная ошибка: {mse}")
    print(f"Квадратичная ошибка: {rmse}")
    print(f"Коэффициент детерминации R^2: {r2}")

    return model, X_test, y_test, y_pred

# Проведение статистических тестов
def statistical_tests(X, y):
    p_values = []
    for col in X.columns:
        correlation, p_value = stats.pearsonr(X[col], y)
        p_values.append((col, correlation, p_value))

    print("\nСтатистическая значимость факторов:")
    for factor, corr, p_val in p_values:
        print(f"Фактор: {factor}, Корреляция: {corr:.4f}, P-значение: {p_val:.4f}")

# Прогнозирование с новыми данными
def predict_new_data(model, new_data):
    prediction = model.predict(new_data)
    print("\nПрогнозы для новых данных:")
    print(prediction)

# Основная функция
def main():
    data_file_path = 'C:/Users/Alexander/source/repos/PythonApplication1/PythonApplication1/data/covid_19.csv'

    data = load_data(data_file_path)

    check_correlations(data)

    model, X_test, y_test, y_pred = train_model(data)

    statistical_tests(data[['Confirmed', 'Recovered', 'Active', 'Lat', 'Long']], data['Deaths'])



if __name__ == "__main__":
    main()
