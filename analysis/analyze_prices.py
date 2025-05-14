import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
import logging

def analyze_price_distribution(df: pd.DataFrame) -> None:
    """Анализ распределения цен"""
    
    # Рассчитываем цену за метр для всех записей
    df['calc_price_per_meter'] = df['price'] / df['Общая площадь']
    
    # Базовая статистика
    print("\nБазовая статистика:")
    print("\nИсходные цены:")
    print(df['price'].describe())
    print("\nПлощади:")
    print(df['Общая площадь'].describe())
    print("\nРасчетная цена за метр:")
    print(df['calc_price_per_meter'].describe())
    
    # Группируем по диапазонам цен
    price_ranges = [0, 100, 500, 1000, 5000, 10000, 50000, float('inf')]
    df['price_range'] = pd.cut(df['price'], bins=price_ranges)
    
    print("\nРаспределение по диапазонам цен:")
    price_dist = df['price_range'].value_counts().sort_index()
    print(price_dist)
    
    # Анализируем цены в разных диапазонах
    for price_range in price_dist.index:
        range_df = df[df['price_range'] == price_range]
        print(f"\nАнализ диапазона {price_range}:")
        print(f"Количество записей: {len(range_df)}")
        print("\nСтатистика площадей:")
        print(range_df['Общая площадь'].describe())
        print("\nСтатистика расчетной цены за метр:")
        print(range_df['calc_price_per_meter'].describe())
        
        # Выводим несколько примеров
        print("\nПримеры объявлений:")
        for _, row in range_df.head(3).iterrows():
            print(f"\nЗаголовок: {row['title']}")
            print(f"Цена: {row['price']:,.2f}")
            print(f"Площадь: {row['Общая площадь']} м²")
            print(f"Расчетная цена за метр: {row['calc_price_per_meter']:,.2f}")
            print(f"Адрес: {row['address']}")

def main():
    # Загружаем данные
    df = pd.read_csv('scrapers/merged_rent_data.csv')
    
    # Конвертируем площадь в числовой формат
    df['Общая площадь'] = pd.to_numeric(df['Общая площадь'].str.replace(',', '.').str.extract(r'(\d+(?:\.\d+)?)', expand=False), errors='coerce')
    
    # Очищаем площади от нулевых значений
    df = df[df['Общая площадь'] > 0]
    
    # Анализируем распределение
    analyze_price_distribution(df)

if __name__ == "__main__":
    main() 