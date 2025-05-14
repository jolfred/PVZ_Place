import matplotlib
matplotlib.use('Agg')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Настройка отображения графиков
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

# Read the data
df = pd.read_csv('scrapers/merged_rent_data.csv')

# Basic data cleaning
df['price'] = pd.to_numeric(df['price'], errors='coerce')
df['Общая площадь'] = df['Общая площадь'].str.extract('(\d+\.?\d*)').astype(float)
df['price_per_sqm'] = df['price'] / df['Общая площадь']

# Создаем директорию для графиков, если её нет
if not os.path.exists('plots'):
    os.makedirs('plots')

# Визуализация цены аренды
plt.figure(figsize=(15, 10))

# 1. Гистограмма цены аренды
plt.subplot(2, 2, 1)
sns.histplot(data=df, x='price', bins=50)
plt.title('Распределение цены аренды')
plt.xlabel('Цена (руб.)')
plt.ylabel('Количество')

# 2. Боксплот цены аренды
plt.subplot(2, 2, 2)
sns.boxplot(data=df, y='price')
plt.title('Боксплот цены аренды')
plt.ylabel('Цена (руб.)')

# 3. Гистограмма цены за квадратный метр
plt.subplot(2, 2, 3)
sns.histplot(data=df, x='price_per_sqm', bins=50)
plt.title('Распределение цены за квадратный метр')
plt.xlabel('Цена за м² (руб.)')
plt.ylabel('Количество')

# 4. Боксплот цены за квадратный метр
plt.subplot(2, 2, 4)
sns.boxplot(data=df, y='price_per_sqm')
plt.title('Боксплот цены за квадратный метр')
plt.ylabel('Цена за м² (руб.)')

plt.tight_layout()
plt.savefig('plots/price_analysis.png')
plt.close()

# Проверяем конкретный адрес
specific_address = df[df['address'].str.contains('Шамиля Усманова, 18Д', na=False)]
print("\nИнформация по адресу ул. Шамиля Усманова, 18Д:")
for _, row in specific_address.iterrows():
    print(f"URL: {row['item_url']}")
    print(f"Цена за м²: {row['price_per_sqm']:.2f}")
    print(f"Общая цена: {row['price']}")
    print(f"Площадь: {row['Общая площадь']}")
    print(f"Название: {row['title']}")
    print("-" * 80)

# --- Поиск выбросов по цене за квадратный метр ---
q1 = df['price_per_sqm'].quantile(0.25)
q3 = df['price_per_sqm'].quantile(0.75)
iqr = q3 - q1
outlier_min = q1 - 1.5 * iqr
outlier_max = q3 + 1.5 * iqr

print("\nСтатистика по цене за квадратный метр:")
print(f"Q1 (25-й процентиль): {q1:.2f}")
print(f"Q3 (75-й процентиль): {q3:.2f}")
print(f"IQR: {iqr:.2f}")
print(f"Нижняя граница выбросов: {outlier_min:.2f}")
print(f"Верхняя граница выбросов: {outlier_max:.2f}")

# Находим выбросы с высокими ценами
high_price_outliers = df[df['price_per_sqm'] > outlier_max]

print("\nВыбросы с высокими ценами за квадратный метр:")
print(f"Всего выбросов с высокими ценами: {len(high_price_outliers)}")
print("\nПримеры выбросов с высокими ценами:")
for _, row in high_price_outliers.iterrows():
    print(f"URL: {row['item_url']}")
    print(f"Цена за м²: {row['price_per_sqm']:.2f}")
    print(f"Общая цена: {row['price']}")
    print(f"Площадь: {row['Общая площадь']}")
    print(f"Название: {row['title']}")
    print("-" * 80)

# Находим выбросы с низкими ценами
low_price_outliers = df[df['price_per_sqm'] < outlier_min]

print("\nВыбросы с низкими ценами за квадратный метр:")
print(f"Всего выбросов с низкими ценами: {len(low_price_outliers)}")
print("\nПримеры выбросов с низкими ценами:")
for _, row in low_price_outliers.iterrows():
    print(f"URL: {row['item_url']}")
    print(f"Цена за м²: {row['price_per_sqm']:.2f}")
    print(f"Общая цена: {row['price']}")
    print(f"Площадь: {row['Общая площадь']}")
    print(f"Название: {row['title']}")
    print("-" * 80) 