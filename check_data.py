import pandas as pd

# Загружаем данные
rent_data = pd.read_csv('scrapers/cleaned_rent_data.csv')
sale_data = pd.read_csv('scrapers/cleaned_sale_data.csv')

# Выводим информацию о колонках
print("Columns in rent_data:")
print(rent_data.columns.tolist())
print("\nColumns in sale_data:")
print(sale_data.columns.tolist())

# Выводим первые несколько строк
print("\nFirst few rows of rent_data:")
print(rent_data.head())
print("\nFirst few rows of sale_data:")
print(sale_data.head()) 