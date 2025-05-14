import pandas as pd
import numpy as np
import logging
from datetime import datetime
import os
import sys

# Создаем директорию для логов, если она не существует
os.makedirs('logs', exist_ok=True)

# Настройка логирования с указанием кодировки UTF-8
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/outliers_urls_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

def is_per_meter_rental(title: str, price: float, area: float) -> bool:
    """
    Определяет, является ли объявление арендой за квадратный метр
    
    Args:
        title (str): Заголовок объявления
        price (float): Цена
        area (float): Площадь

    Returns:
        bool: True если это аренда за м², False если общая аренда
    """
    # Проверяем ключевые слова в заголовке
    keywords = ['за м²', 'за м2', 'за кв.м', 'за кв м', 'за квадратный метр', 'кв.м.', 'кв. м.', 'м2/мес', 'м²/мес']
    if any(keyword.lower() in title.lower() for keyword in keywords):
        return True
    
    # Рассчитываем предполагаемую цену за м²
    price_per_m2 = price / area if area > 0 else float('inf')
    
    # Если цена за м² получается нереально низкой (меньше 100 руб) или 
    # нереально высокой (больше 50000 руб), то это скорее всего общая цена
    if price_per_m2 < 100 or price_per_m2 > 50000:
        return False
    
    # Если цена меньше 1000 рублей И при этом цена за м² получается разумной,
    # то это вероятно цена за м²
    if price < 1000 and 200 <= price_per_m2 <= 7000:
        return True
    
    # Если общая цена при пересчете дает слишком низкую цену за метр,
    # то это вероятно цена за м²
    if 200 <= price <= 7000 and (price * area) <= 1000000:
        return True
        
    return False

def clean_price(row) -> float:
    """
    Очистка цены. Определяет тип аренды и корректирует цену соответственно.
    
    Args:
        row: Строка DataFrame с данными объявления

    Returns:
        float: Очищенная цена аренды за месяц
    """
    price = row['price']
    area = row['Общая площадь']
    title = row['title']
    
    # Определяем тип аренды
    if is_per_meter_rental(title, price, area):
        # Если это цена за м², умножаем на площадь
        return price * area
    
    # Иначе это общая цена аренды
    return price

def is_commercial_property(title: str, address: str) -> bool:
    """
    Определяет, является ли объявление коммерческой недвижимостью
    
    Args:
        title (str): Заголовок объявления
        address (str): Адрес объекта

    Returns:
        bool: True если это коммерческая недвижимость
    """
    commercial_keywords = [
        'офис', 'торгов', 'коммерческ', 'магазин', 'склад', 'производств',
        'помещение', 'бизнес', 'свободного назначения', 'псн', 'юридическ',
        'нежил', 'аренда помещ'
    ]
    
    text_to_check = (title + ' ' + address).lower()
    return any(keyword.lower() in text_to_check for keyword in commercial_keywords)

def get_property_type(title: str, address: str) -> str:
    """
    Определяет тип помещения на основе заголовка и адреса
    
    Args:
        title (str): Заголовок объявления
        address (str): Адрес объекта

    Returns:
        str: Тип помещения ('premium', 'standard' или 'basic')
    """
    text = (title + ' ' + address).lower()
    
    # Премиум помещения (офисы в центре, торговые помещения и т.д.)
    premium_keywords = ['офис', 'торгов', 'магазин', 'бутик', 'салон', 
                       'общепит', 'ресторан', 'кафе', 'медицин']
    
    # Стандартные помещения
    standard_keywords = ['помещение', 'свободного назначения', 'псн']
    
    # Базовые помещения (склады, контейнеры и т.д.)
    basic_keywords = ['склад', 'контейнер', 'производств', 'гараж', 
                     'бокс', 'ангар', 'пром', 'цех']
    
    # Сначала проверяем базовые помещения, так как они более специфичны
    if any(keyword in text for keyword in basic_keywords):
        return 'basic'
    elif any(keyword in text for keyword in premium_keywords):
        return 'premium'
    else:
        return 'standard'

def find_price_outliers(df: pd.DataFrame) -> tuple:
    """Поиск выбросов в ценах"""
    # Сначала корректируем цены в зависимости от типа аренды
    df['price'] = df.apply(clean_price, axis=1)
    
    # Определяем тип помещения
    df['property_type'] = df.apply(lambda row: get_property_type(row['title'], row['address']), axis=1)
    
    # Рассчитываем цены за квадратный метр
    df['price_per_m2'] = df['price'] / df['Общая площадь']
    
    # Отфильтровываем явно ошибочные цены (меньше 1000 рублей в месяц)
    absolute_min_price = 1000  # Минимальная месячная аренда
    df = df[df['price'] >= absolute_min_price].copy()
    
    # Находим выбросы отдельно для каждого типа помещений
    outliers = {'low': [], 'high': [], 'thresholds': {}}
    
    for prop_type in ['premium', 'standard', 'basic']:
        df_segment = df[df['property_type'] == prop_type]
        if len(df_segment) > 0:
            # Рассчитываем квартили для цен за м²
            Q1 = df_segment['price_per_m2'].quantile(0.25)
            Q3 = df_segment['price_per_m2'].quantile(0.75)
            IQR = Q3 - Q1
            
            # Определяем границы в зависимости от типа помещения
            if prop_type == 'premium':
                lower_bound = max(700, Q1 - 2.0 * IQR)
                upper_bound = min(20000, Q3 + 2.0 * IQR)
            elif prop_type == 'basic':
                lower_bound = max(200, Q1 - 2.0 * IQR)
                upper_bound = min(7000, Q3 + 2.0 * IQR)
            else:  # standard
                lower_bound = max(500, Q1 - 2.0 * IQR)
                upper_bound = min(15000, Q3 + 2.0 * IQR)
            
            # Сохраняем пороговые значения
            outliers['thresholds'][prop_type] = (lower_bound, upper_bound)
            
            # Находим выбросы
            outliers['low'].append(df_segment[df_segment['price_per_m2'] < lower_bound])
            outliers['high'].append(df_segment[df_segment['price_per_m2'] > upper_bound])
    
    # Объединяем результаты
    price_outliers_low = pd.concat(outliers['low']) if outliers['low'] else pd.DataFrame()
    price_outliers_high = pd.concat(outliers['high']) if outliers['high'] else pd.DataFrame()
    
    return price_outliers_low, price_outliers_high, outliers['thresholds']

def print_outlier_info(df: pd.DataFrame, title: str, outlier_type: str, thresholds: dict):
    """Вывод информации о выбросах"""
    logging.info(f"\n{title}:")
    for _, row in df.iterrows():
        logging.info("\nИнформация об объявлении:")
        logging.info(f"URL: {row['item_url']}")
        logging.info(f"Заголовок: {row['title']}")
        logging.info(f"Адрес: {row['address']}")
        
        # Определяем тип помещения
        property_type = get_property_type(row['title'], row['address'])
        property_type_names = {
            'premium': 'Премиум помещение (офис/торговля)',
            'standard': 'Стандартное помещение',
            'basic': 'Базовое помещение (склад/контейнер)'
        }
        logging.info(f"Тип помещения: {property_type_names[property_type]}")
        
        # Определяем тип аренды
        is_per_meter = is_per_meter_rental(row['title'], row['price'], row['Общая площадь'])
        original_price = row['price']
        actual_price = clean_price(row)
        
        if is_per_meter:
            logging.info(f"Тип аренды: За квадратный метр")
            logging.info(f"Цена за м²: {original_price:,.2f} руб.")
            logging.info(f"Общая цена: {actual_price:,.2f} руб.")
        else:
            logging.info(f"Тип аренды: Общая стоимость")
            logging.info(f"Цена: {actual_price:,.2f} руб.")
            
        logging.info(f"Площадь: {row['Общая площадь']:.2f} м2")
        price_per_m2 = actual_price / row['Общая площадь']
        logging.info(f"Итоговая цена за м2: {price_per_m2:,.2f} руб.")
        
        # Получаем пороговые значения для данного типа помещения
        lower_bound, upper_bound = thresholds[property_type]
        
        # Добавляем информацию о причине выброса
        if outlier_type == 'price_low':
            logging.info(f"Причина: Итоговая цена за м² ({price_per_m2:,.2f} руб.) ниже допустимого порога в {lower_bound:,.2f} руб. для типа помещения '{property_type_names[property_type]}'")
        elif outlier_type == 'price_high':
            logging.info(f"Причина: Итоговая цена за м² ({price_per_m2:,.2f} руб.) выше допустимого порога в {upper_bound:,.2f} руб. для типа помещения '{property_type_names[property_type]}'")
        
        logging.info("-" * 80)

def main():
    # Загружаем данные
    df = pd.read_csv('scrapers/merged_rent_data.csv')
    
    # Конвертируем строковые значения в числовые
    df['Общая площадь'] = pd.to_numeric(df['Общая площадь'].str.replace('м²', '').str.strip(), errors='coerce')
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    
    # Удаляем строки с отсутствующими значениями
    df = df.dropna(subset=['Общая площадь', 'price'])
    
    logging.info(f"Загружено {len(df)} записей")
    
    # Анализируем выбросы в ценах
    price_outliers_low, price_outliers_high, thresholds = find_price_outliers(df)
    logging.info(f"\nНайдено выбросов в ценах:")
    logging.info(f"Низкие цены: {len(price_outliers_low)} записей")
    logging.info(f"Высокие цены: {len(price_outliers_high)} записей")
    
    logging.info(f"\nПороговые значения для цен за м² по типам помещений:")
    for prop_type, (lower, upper) in thresholds.items():
        if prop_type == 'premium':
            type_name = "Премиум помещения (офисы/торговля)"
        elif prop_type == 'basic':
            type_name = "Базовые помещения (склады/контейнеры)"
        else:
            type_name = "Стандартные помещения"
        logging.info(f"{type_name}:")
        logging.info(f"  Минимальная цена: {lower:,.2f} руб.")
        logging.info(f"  Максимальная цена: {upper:,.2f} руб.")
    
    # Выводим детальную информацию о выбросах с указанием причин
    print_outlier_info(price_outliers_low, "Объявления с подозрительно низкими ценами за м2", 'price_low', thresholds)
    print_outlier_info(price_outliers_high, "Объявления с подозрительно высокими ценами за м2", 'price_high', thresholds)

if __name__ == "__main__":
    main() 