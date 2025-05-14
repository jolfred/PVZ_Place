import pandas as pd
import numpy as np
import re
from typing import Tuple, List, Dict
import logging
from datetime import datetime
import os
import sys

# Создаем директорию для логов, если она не существует
os.makedirs('logs', exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/data_cleaning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)

class DataCleaner:
    def __init__(self, input_file: str, property_type: str = 'rent'):
        """
        Инициализация очистки данных
        
        Args:
            input_file (str): Путь к входному CSV файлу
            property_type (str): Тип объявлений ('rent' или 'sale')
        """
        self.input_file = input_file
        self.property_type = property_type
        self.df = pd.read_csv(input_file)
        logging.info(f"Загружено {len(self.df)} записей из {input_file}")
        
        # Устанавливаем пороговые значения в зависимости от типа
        if property_type == 'rent':
            self.min_total_price = 1000  # минимальная общая цена аренды
            self.min_price_per_meter = 100  # минимальная цена за м² аренды
            self.max_price_per_meter = 5000  # максимальная цена за м² аренды
            self.max_total_price = 500000  # максимальная общая цена аренды
        else:  # sale
            self.min_total_price = 100000  # минимальная цена продажи
            self.min_price_per_meter = 10000  # минимальная цена за м² продажи
            self.max_price_per_meter = 300000  # максимальная цена за м² продажи
            self.max_total_price = 50000000  # максимальная цена продажи

    def extract_city(self, address: str) -> str:
        """Извлечение города из адреса"""
        match = re.search(r'([А-ЯЁ][а-яё]+(?: [А-ЯЁ][а-яё]+)*)$', address)
        return match.group(1) if match else ""
    
    def extract_street(self, address: str, city: str) -> str:
        """Извлечение улицы из адреса"""
        if city and address.endswith(city):
            address = address[:-(len(city))]
        address = re.sub(r'[\s,]*\d+[\w/А-Яа-я]*$', '', address)
        return address.strip(' ,')
    
    def is_per_meter_price(self, title: str, price: float, area: float) -> bool:
        """
        Определяет, является ли цена за квадратный метр
        
        Args:
            title (str): Заголовок объявления
            price (float): Цена
            area (float): Площадь

        Returns:
            bool: True если это цена за м², False если общая цена
        """
        # Проверяем ключевые слова в заголовке
        keywords = ['за м²', 'за м2', 'за кв.м', 'за кв м', 'за квадратный метр', 'кв.м.', 'кв. м.', 'м2/мес', 'м²/мес', 'руб/м²', 'руб/м2']
        if any(keyword.lower() in title.lower() for keyword in keywords):
            return True
            
        # Проверяем соотношение цены к площади
        if self.property_type == 'rent':
            # Если цена меньше 1000 рублей и площадь больше 10м², 
            # это почти наверняка цена за м²
            if price < 1000 and area > 10:
                return True
                
            # Проверяем расчетную цену за метр
            calculated_price_per_m2 = price / area if area > 0 else float('inf')
            
            # Если получается адекватная цена за метр и общая цена нереалистично низкая
            if 100 <= calculated_price_per_m2 <= 3000 and price < 5000:
                return True
                
        else:  # sale
            if price < 50000 and area > 10:
                return True
            calculated_price_per_m2 = price / area if area > 0 else float('inf')
            if 10000 <= calculated_price_per_m2 <= 100000 and price < 1000000:
                return True
                
        return False

    def clean_area(self, area_str: str) -> float:
        """Очистка площади"""
        try:
            area = float(re.sub(r'[^\d.]', '', str(area_str)))
            return area if area > 0 else np.nan
        except:
            return np.nan

    def calculate_price_metrics(self, price: float, area: float, title: str) -> Tuple[float, float, bool]:
        """
        Рассчитывает метрики цены: за метр и общую
        
        Args:
            price (float): Исходная цена
            area (float): Площадь помещения
            title (str): Заголовок объявления
        
        Returns:
            Tuple[float, float, bool]: (общая цена, цена за м², является_ли_ценой_за_метр)
        """
        # Проверяем ключевые слова для цены за метр
        keywords = ['за м²', 'за м2', 'за кв.м', 'за кв м', 'за квадратный метр', 
                   'кв.м.', 'кв. м.', 'м2/мес', 'м²/мес', 'руб/м²', 'руб/м2']
        is_per_meter = any(keyword.lower() in title.lower() for keyword in keywords)
        
        # Рассчитываем обе метрики цены
        if is_per_meter:
            price_per_meter = price
            total_price = price * area
        else:
            # Если цена слишком низкая для общей, считаем что это за метр
            if price < 1000 and area >= 10:
                is_per_meter = True
                price_per_meter = price
                total_price = price * area
            else:
                total_price = price
                price_per_meter = price / area if area > 0 else float('inf')
                
                # Если получившаяся цена за метр выглядит более реалистично чем общая
                if 100 <= price_per_meter <= 3000 and total_price < 5000:
                    is_per_meter = True
                    price_per_meter = total_price / area
                    total_price = price * area
        
        return total_price, price_per_meter, is_per_meter

    def clean_price(self, row) -> Tuple[float, float]:
        """
        Очистка цены
        
        Args:
            row: Строка DataFrame с данными объявления

        Returns:
            Tuple[float, float]: (общая цена, цена за м²)
        """
        price = row['price']
        area = row['Общая площадь']
        title = row['title']
        
        # Проверяем на placeholder цены (1 рубль)
        if price == 1:
            return np.nan, np.nan
        
        # Рассчитываем метрики цены
        total_price, price_per_meter, is_per_meter = self.calculate_price_metrics(price, area, title)
        
        # Проверяем на адекватность цены за метр
        if price_per_meter < self.min_price_per_meter:
            logging.info(f"\nСлишком низкая цена за м² ({price_per_meter:,.2f} < {self.min_price_per_meter:,.2f}):")
            logging.info(f"Заголовок: {title}")
            logging.info(f"Адрес: {row['address']}")
            logging.info(f"Площадь: {area} м²")
            logging.info(f"Исходная цена: {price:,.2f}")
            logging.info(f"Определено как: {'Цена за м²' if is_per_meter else 'Общая цена'}")
            return np.nan, np.nan
            
        if price_per_meter > self.max_price_per_meter:
            logging.info(f"\nСлишком высокая цена за м² ({price_per_meter:,.2f} > {self.max_price_per_meter:,.2f}):")
            logging.info(f"Заголовок: {title}")
            logging.info(f"Адрес: {row['address']}")
            logging.info(f"Площадь: {area} м²")
            logging.info(f"Исходная цена: {price:,.2f}")
            logging.info(f"Определено как: {'Цена за м²' if is_per_meter else 'Общая цена'}")
            return np.nan, np.nan
        
        return total_price, price_per_meter

    def remove_price_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Удаление выбросов в ценах"""
        # Фильтруем явно ошибочные цены
        df = df[
            (df['total_price'] >= self.min_total_price) & 
            (df['price_per_meter'] >= self.min_price_per_meter) &
            (df['total_price'] <= self.max_total_price) &
            (df['price_per_meter'] <= self.max_price_per_meter)
        ].copy()
        
        # Рассчитываем квартили отдельно для каждого типа цены
        Q1_total = df['total_price'].quantile(0.25)
        Q3_total = df['total_price'].quantile(0.75)
        IQR_total = Q3_total - Q1_total
        
        Q1_per_meter = df['price_per_meter'].quantile(0.25)
        Q3_per_meter = df['price_per_meter'].quantile(0.75)
        IQR_per_meter = Q3_per_meter - Q1_per_meter
        
        # Определяем границы для обоих типов цен
        total_price_lower = Q1_total - 1.5 * IQR_total
        total_price_upper = Q3_total + 1.5 * IQR_total
        
        per_meter_lower = Q1_per_meter - 1.5 * IQR_per_meter
        per_meter_upper = Q3_per_meter + 1.5 * IQR_per_meter
        
        # Фильтруем выбросы
        df = df[
            (df['total_price'] >= total_price_lower) & 
            (df['total_price'] <= total_price_upper) &
            (df['price_per_meter'] >= per_meter_lower) & 
            (df['price_per_meter'] <= per_meter_upper)
        ]
        
        return df
    
    def extract_area_from_title(self, title: str) -> float:
        """
        Извлекает площадь из заголовка, если она указана в формате 'от X до Y м²'
        
        Args:
            title (str): Заголовок объявления
        
        Returns:
            float: Минимальная площадь из диапазона или None если не найдена
        """
        # Ищем паттерн "от X до Y м²"
        pattern = r'от (\d+(?:\.\d+)?)(?: до (\d+(?:\.\d+)?))? ?м²'
        match = re.search(pattern, title)
        if match:
            min_area = float(match.group(1))
            return min_area
        return None

    def clean_area(self, area_str: str, title: str = "") -> float:
        """
        Очистка площади с учетом заголовка
        
        Args:
            area_str: Строка с площадью
            title: Заголовок объявления
        
        Returns:
            float: Очищенное значение площади
        """
        # Сначала пытаемся получить площадь из основного поля
        try:
            area = float(re.sub(r'[^\d.]', '', str(area_str)))
            if area > 0:
                return area
        except:
            pass
        
        # Если не получилось, пытаемся извлечь из заголовка
        if title:
            area_from_title = self.extract_area_from_title(title)
            if area_from_title is not None:
                return area_from_title
        
        return np.nan

    def _similarity(self, s1: str, s2: str) -> float:
        """Вычисляет схожесть двух строк"""
        if not s1 or not s2:
            return 0.0
        
        s1 = s1.lower()
        s2 = s2.lower()
        
        # Простая метрика схожести на основе общих слов
        words1 = set(s1.split())
        words2 = set(s2.split())
        
        common_words = words1.intersection(words2)
        all_words = words1.union(words2)
        
        if not all_words:
            return 0.0
            
        return len(common_words) / len(all_words)

    def find_duplicates(self, similarity_threshold: float = 0.8) -> List[Tuple[int, int]]:
        """
        Находит дубликаты объявлений
        
        Args:
            similarity_threshold: Порог схожести для определения дубликатов
            
        Returns:
            List[Tuple[int, int]]: Список пар индексов дубликатов
        """
        duplicates = []
        n = len(self.df)
        
        for i in range(n):
            for j in range(i + 1, n):
                # Проверяем схожесть заголовков и адресов
                title_sim = self._similarity(str(self.df.iloc[i]['title']), str(self.df.iloc[j]['title']))
                addr_sim = self._similarity(str(self.df.iloc[i]['address']), str(self.df.iloc[j]['address']))
                
                # Проверяем схожесть цен и площадей
                price_match = abs(self.df.iloc[i]['price'] - self.df.iloc[j]['price']) < 1
                area_match = abs(self.df.iloc[i]['Общая площадь'] - self.df.iloc[j]['Общая площадь']) < 0.1
                
                # Если объявления похожи по всем параметрам
                if ((title_sim + addr_sim) / 2 >= similarity_threshold and 
                    price_match and area_match):
                    duplicates.append((i, j))
        
        return duplicates

    def clean_data(self) -> pd.DataFrame:
        """
        Основной метод очистки данных
        
        Returns:
            pd.DataFrame: Очищенный DataFrame
        """
        logging.info("Начинаем очистку данных...")
        
        # Удаляем дубликаты
        duplicates = self.find_duplicates()
        duplicate_indices = set([j for _, j in duplicates])  # Оставляем первое появление
        self.df = self.df.drop(index=duplicate_indices)
        logging.info(f"Удалено {len(duplicate_indices)} дубликатов")
        
        # Очищаем адреса
        self.df['city'] = self.df['address'].apply(self.extract_city)
        self.df['street'] = self.df.apply(lambda row: self.extract_street(row['address'], row['city']), axis=1)
        
        # Очищаем площади
        self.df['Общая площадь'] = self.df.apply(
            lambda row: self.clean_area(row['Общая площадь'], row['title']), 
            axis=1
        )
        
        # Очищаем цены
        price_data = self.df.apply(self.clean_price, axis=1, result_type='expand')
        self.df['total_price'] = price_data[0]
        self.df['price_per_meter'] = price_data[1]
        
        # Удаляем записи с отсутствующими важными данными
        self.df = self.df.dropna(subset=['total_price', 'price_per_meter', 'Общая площадь'])
        
        # Удаляем выбросы в ценах
        self.df = self.remove_price_outliers(self.df)
        
        logging.info(f"Очистка завершена. Осталось {len(self.df)} записей")
        return self.df

    def save_results(self, output_file: str):
        """Сохранение результатов в CSV"""
        self.df.to_csv(output_file, index=False, encoding='utf-8')
        logging.info(f"Результаты сохранены в {output_file}")

    def generate_report(self) -> Dict:
        """
        Генерирует отчет о данных
        
        Returns:
            Dict: Словарь с метриками
        """
        return {
            'total_records': len(self.df),
            'cities': self.df['city'].value_counts().to_dict(),
            'total_price_stats': {
                'min': self.df['total_price'].min(),
                'max': self.df['total_price'].max(),
                'mean': self.df['total_price'].mean(),
                'median': self.df['total_price'].median()
            },
            'price_per_meter_stats': {
                'min': self.df['price_per_meter'].min(),
                'max': self.df['price_per_meter'].max(),
                'mean': self.df['price_per_meter'].mean(),
                'median': self.df['price_per_meter'].median()
            },
            'area_stats': {
                'min': self.df['Общая площадь'].min(),
                'max': self.df['Общая площадь'].max(),
                'mean': self.df['Общая площадь'].mean(),
                'median': self.df['Общая площадь'].median()
            }
        }

    def print_extreme_prices(self):
        """Выводит информацию о экстремальных ценах"""
        # Самые дорогие объявления
        logging.info("\nТоп-5 самых дорогих объявлений:")
        for _, row in self.df.nlargest(5, 'total_price').iterrows():
            logging.info(f"\nЗаголовок: {row['title']}")
            logging.info(f"Адрес: {row['address']}")
            logging.info(f"Площадь: {row['Общая площадь']} м²")
            logging.info(f"Общая цена: {row['total_price']:,.2f}")
            logging.info(f"Цена за м²: {row['price_per_meter']:.2f}")
        
        # Самые дешевые объявления
        logging.info("\nТоп-5 самых дешевых объявлений:")
        for _, row in self.df.nsmallest(5, 'total_price').iterrows():
            logging.info(f"\nЗаголовок: {row['title']}")
            logging.info(f"Адрес: {row['address']}")
            logging.info(f"Площадь: {row['Общая площадь']} м²")
            logging.info(f"Общая цена: {row['total_price']:,.2f}")
            logging.info(f"Цена за м²: {row['price_per_meter']:.2f}") 