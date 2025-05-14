import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
import os

# Настройка стиля графиков
plt.style.use('seaborn')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Создаем директории для логов и графиков
os.makedirs('logs', exist_ok=True)
os.makedirs('plots', exist_ok=True)

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/data_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)

class RentalAnalyzer:
    def __init__(self, input_file: str):
        """Инициализация анализатора данных"""
        self.df = pd.read_csv(input_file)
        
        # Конвертируем строковые значения в числовые
        self.df['Общая площадь'] = pd.to_numeric(self.df['Общая площадь'].str.replace('м²', '').str.strip(), errors='coerce')
        self.df['price'] = pd.to_numeric(self.df['price'], errors='coerce')
        
        # Удаляем строки с отсутствующими значениями
        self.df = self.df.dropna(subset=['Общая площадь', 'price'])
        
        logging.info(f"Загружено {len(self.df)} записей из {input_file}")
        
    def analyze_prices(self):
        """Анализ цен"""
        logging.info("\nАнализ цен:")
        
        # Статистика по ценам
        price_stats = self.df['price'].describe()
        logging.info("\nСтатистика по ценам:")
        for stat, value in price_stats.items():
            logging.info(f"{stat}: {value:,.2f}")
        
        # Распределение цен
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='price', bins=50)
        plt.title('Распределение цен')
        plt.xlabel('Цена (руб.)')
        plt.ylabel('Количество объявлений')
        plt.savefig('plots/price_distribution.png')
        plt.close()
        
        # Цены по городам
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='city', y='price')
        plt.xticks(rotation=45)
        plt.title('Распределение цен по городам')
        plt.xlabel('Город')
        plt.ylabel('Цена (руб.)')
        plt.tight_layout()
        plt.savefig('plots/price_by_city.png')
        plt.close()
        
    def analyze_areas(self):
        """Анализ площадей"""
        logging.info("\nАнализ площадей:")
        
        # Статистика по площадям
        area_stats = self.df['Общая площадь'].describe()
        logging.info("\nСтатистика по площадям:")
        for stat, value in area_stats.items():
            logging.info(f"{stat}: {value:.2f}")
        
        # Распределение площадей
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='Общая площадь', bins=50)
        plt.title('Распределение площадей')
        plt.xlabel('Площадь (м²)')
        plt.ylabel('Количество объявлений')
        plt.savefig('plots/area_distribution.png')
        plt.close()
        
        # Площади по городам
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='city', y='Общая площадь')
        plt.xticks(rotation=45)
        plt.title('Распределение площадей по городам')
        plt.xlabel('Город')
        plt.ylabel('Площадь (м²)')
        plt.tight_layout()
        plt.savefig('plots/area_by_city.png')
        plt.close()
        
    def analyze_price_per_meter(self):
        """Анализ цен за квадратный метр"""
        self.df['price_per_m2'] = self.df['price'] / self.df['Общая площадь']
        logging.info("\nАнализ цен за квадратный метр:")
        
        # Статистика по ценам за метр
        price_m2_stats = self.df['price_per_m2'].describe()
        logging.info("\nСтатистика по ценам за м²:")
        for stat, value in price_m2_stats.items():
            logging.info(f"{stat}: {value:,.2f}")
        
        # Распределение цен за метр
        plt.figure(figsize=(12, 6))
        sns.histplot(data=self.df, x='price_per_m2', bins=50)
        plt.title('Распределение цен за квадратный метр')
        plt.xlabel('Цена за м² (руб.)')
        plt.ylabel('Количество объявлений')
        plt.savefig('plots/price_per_m2_distribution.png')
        plt.close()
        
        # Цены за метр по городам
        plt.figure(figsize=(12, 6))
        sns.boxplot(data=self.df, x='city', y='price_per_m2')
        plt.xticks(rotation=45)
        plt.title('Распределение цен за м² по городам')
        plt.xlabel('Город')
        plt.ylabel('Цена за м² (руб.)')
        plt.tight_layout()
        plt.savefig('plots/price_per_m2_by_city.png')
        plt.close()
        
    def analyze_correlations(self):
        """Анализ корреляций"""
        # Корреляция между ценой и площадью
        plt.figure(figsize=(10, 6))
        sns.scatterplot(data=self.df, x='Общая площадь', y='price', alpha=0.5)
        plt.title('Зависимость цены от площади')
        plt.xlabel('Площадь (м²)')
        plt.ylabel('Цена (руб.)')
        plt.savefig('plots/price_area_correlation.png')
        plt.close()
        
        # Добавляем линию тренда
        plt.figure(figsize=(10, 6))
        sns.regplot(data=self.df, x='Общая площадь', y='price', scatter_kws={'alpha':0.5})
        plt.title('Зависимость цены от площади с линией тренда')
        plt.xlabel('Площадь (м²)')
        plt.ylabel('Цена (руб.)')
        plt.savefig('plots/price_area_correlation_with_trend.png')
        plt.close()

def main():
    # Создаем анализатор
    analyzer = RentalAnalyzer('scrapers/cleaned_rent_data.csv')
    
    # Проводим анализ
    analyzer.analyze_prices()
    analyzer.analyze_areas()
    analyzer.analyze_price_per_meter()
    analyzer.analyze_correlations()
    
    logging.info("\nАнализ завершен. Графики сохранены в директории 'plots'")

if __name__ == "__main__":
    main() 