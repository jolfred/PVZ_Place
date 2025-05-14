import pandas as pd
import re
from typing import List, Tuple
from difflib import SequenceMatcher
import logging
from datetime import datetime
import os

# Создаем директорию для логов, если она не существует
os.makedirs('logs', exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/duplicate_urls_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def similarity(s1: str, s2: str) -> float:
    """Вычисление схожести строк"""
    return SequenceMatcher(None, s1, s2).ratio()

def extract_city(address: str) -> str:
    """Извлекает название города из адреса"""
    # Удаляем пробелы в начале и конце
    address = address.strip()
    # Находим последнее слово в адресе (обычно это город)
    return address.split(',')[-1].strip() if ',' in address else address.split()[-1]

def find_duplicates(df: pd.DataFrame, similarity_threshold: float = 0.8) -> List[Tuple[int, int]]:
    """Поиск дубликатов с предварительной фильтрацией"""
    duplicates = []
    
    # Создаем словарь для группировки объявлений по городу, цене и площади
    groups = {}
    for i in range(len(df)):
        price = df.iloc[i]['price']
        area = df.iloc[i]['Общая площадь']
        city = extract_city(df.iloc[i]['address'])
        
        # Округляем цену и площадь для группировки похожих значений
        price_group = round(price, -2)  # округление до сотен
        area_group = round(area, 1)     # округление до 0.1
        key = (city, price_group, area_group)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    
    # Проверяем только объявления внутри одной группы (один город, похожие цена и площадь)
    for group_indices in groups.values():
        if len(group_indices) > 1:  # если в группе больше 1 объявления
            for i in range(len(group_indices)):
                idx1 = group_indices[i]
                for j in range(i + 1, len(group_indices)):
                    idx2 = group_indices[j]
                    
                    # Проверяем схожесть цен и площадей
                    price_sim = abs(df.iloc[idx1]['price'] - df.iloc[idx2]['price']) / max(df.iloc[idx1]['price'], df.iloc[idx2]['price'])
                    area_sim = abs(df.iloc[idx1]['Общая площадь'] - df.iloc[idx2]['Общая площадь']) / max(df.iloc[idx1]['Общая площадь'], df.iloc[idx2]['Общая площадь'])
                    
                    if price_sim < 0.1 and area_sim < 0.1:
                        # Проверяем схожесть адресов (должны быть очень похожи, так как это один город)
                        address_sim = similarity(df.iloc[idx1]['address'], df.iloc[idx2]['address'])
                        if address_sim > 0.9:  # увеличиваем порог для адресов
                            # Только если адреса очень похожи, проверяем заголовок
                            title_sim = similarity(df.iloc[idx1]['title'], df.iloc[idx2]['title'])
                            if title_sim > similarity_threshold:
                                duplicates.append((idx1, idx2))
    
    return duplicates

def main():
    # Загружаем данные
    df = pd.read_csv('scrapers/merged_rent_data.csv')
    
    # Конвертируем строковые значения в числовые
    df['Общая площадь'] = pd.to_numeric(df['Общая площадь'].str.replace('м²', '').str.strip(), errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Удаляем строки с отсутствующими значениями
    df = df.dropna(subset=['Общая площадь', 'price'])
    
    logging.info(f"Загружено {len(df)} записей")
    
    # Ищем дубликаты
    duplicates = find_duplicates(df)
    logging.info(f"Найдено {len(duplicates)} пар дубликатов")
    
    # Выводим информацию о дубликатах
    logging.info("\nПары дубликатов:")
    for i, j in duplicates:
        logging.info("\nПара дубликатов:")
        logging.info(f"1. URL: {df.iloc[i]['item_url']}")
        logging.info(f"   Заголовок: {df.iloc[i]['title']}")
        logging.info(f"   Адрес: {df.iloc[i]['address']}")
        logging.info(f"   Цена: {df.iloc[i]['price']}")
        logging.info(f"   Площадь: {df.iloc[i]['Общая площадь']}")
        
        logging.info(f"\n2. URL: {df.iloc[j]['item_url']}")
        logging.info(f"   Заголовок: {df.iloc[j]['title']}")
        logging.info(f"   Адрес: {df.iloc[j]['address']}")
        logging.info(f"   Цена: {df.iloc[j]['price']}")
        logging.info(f"   Площадь: {df.iloc[j]['Общая площадь']}")
        logging.info("-" * 80)

if __name__ == "__main__":
    main() 