import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime
import os
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Пути к данным
DATA_DIR = os.path.join('data_cleaning', 'data', 'processed')
RENT_DATA_PATH = os.path.join(DATA_DIR, 'rent_data.csv')
SALE_DATA_PATH = os.path.join(DATA_DIR, 'sale_data.csv')

# Настройка отображения графиков
plt.style.use('seaborn-v0_8')  # Используем встроенный стиль matplotlib
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 12

class RealEstateAnalyzer:
    def __init__(self):
        # Загрузка данных
        self.rent_df = pd.read_csv(RENT_DATA_PATH)
        self.sale_df = pd.read_csv(SALE_DATA_PATH)
        
        # Преобразование цен в числовой формат
        self.rent_df['price'] = pd.to_numeric(self.rent_df['price'], errors='coerce')
        self.sale_df['price'] = pd.to_numeric(self.sale_df['price'], errors='coerce')
        
        # Извлечение площади
        self.rent_df['area'] = self.rent_df['title'].apply(self._extract_area)
        self.sale_df['area'] = self.sale_df['title'].apply(self._extract_area)
        
        # Преобразование дат
        self.rent_df['date'] = pd.to_datetime(self.rent_df['date'], format='%d %B %Y', errors='coerce')
        self.sale_df['date'] = pd.to_datetime(self.sale_df['date'], format='%d %B %Y', errors='coerce')

        # Инициализация кластеризации
        self.n_clusters = 5  # Можно настроить
        self.kmeans = None
        self.scaler = StandardScaler()

    def _extract_area(self, title):
        try:
            import re
            match = re.search(r'(\d+(?:\.\d+)?)\s*м²', title)
            if match:
                return float(match.group(1))
            return None
        except:
            return None

    def get_market_statistics(self):
        """Получение общей статистики по рынку"""
        stats = {
            'total_objects': {
                'sale': len(self.sale_df),
                'rent': len(self.rent_df)
            },
            'avg_price': {
                'sale': self.sale_df['price'].mean(),
                'rent': self.rent_df['price'].mean()
            },
            'price_per_m2': {
                'sale': (self.sale_df['price'] / self.sale_df['area']).mean(),
                'rent': (self.rent_df['price'] / self.rent_df['area']).mean()
            },
            'price_trend': {
                'sale': self._calculate_price_trend(self.sale_df),
                'rent': self._calculate_price_trend(self.rent_df)
            }
        }
        return stats

    def _calculate_price_trend(self, df):
        """Расчет тренда цен за последние 3 месяца"""
        df = df.copy()
        df['month'] = df['date'].dt.to_period('M')
        monthly_avg = df.groupby('month')['price'].mean().tail(3)
        if len(monthly_avg) >= 2:
            return ((monthly_avg.iloc[-1] - monthly_avg.iloc[0]) / monthly_avg.iloc[0] * 100)
        return 0

    def get_city_statistics(self, city_name):
        """Получение статистики по конкретному городу"""
        city_sale = self.sale_df[self.sale_df['address'].str.contains(city_name, case=False, na=False)]
        city_rent = self.rent_df[self.rent_df['address'].str.contains(city_name, case=False, na=False)]

        return {
            'total_objects': {
                'sale': len(city_sale),
                'rent': len(city_rent)
            },
            'avg_price': {
                'sale': city_sale['price'].mean(),
                'rent': city_rent['price'].mean()
            },
            'price_per_m2': {
                'sale': (city_sale['price'] / city_sale['area']).mean(),
                'rent': (city_rent['price'] / city_rent['area']).mean()
            },
            'districts': self._get_districts_stats(city_sale, city_rent)
        }

    def _get_districts_stats(self, sale_df, rent_df):
        """Получение статистики по районам"""
        districts = []
        # Извлекаем районы из адресов (предполагаем, что район указан после запятой)
        sale_df['district'] = sale_df['address'].str.extract(r',\s*([^,]+)(?:,|$)')
        rent_df['district'] = rent_df['address'].str.extract(r',\s*([^,]+)(?:,|$)')

        for district in sale_df['district'].unique():
            if pd.isna(district):
                continue
            
            district_sale = sale_df[sale_df['district'] == district]
            district_rent = rent_df[rent_df['district'] == district]
            
            districts.append({
                'name': district,
                'avg_sale_price': district_sale['price'].mean(),
                'avg_rent_price': district_rent['price'].mean(),
                'objects_count': len(district_sale) + len(district_rent)
            })
        
        return districts

    def get_property_clusters(self):
        """Получение кластеров недвижимости"""
        # Подготовка данных для кластеризации
        features = self.sale_df[['price', 'area']].dropna()
        
        # Нормализация данных
        X = self.scaler.fit_transform(features)
        
        # Кластеризация
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = self.kmeans.fit_predict(X)
        
        # Подготовка результатов
        result = []
        for i in range(self.n_clusters):
            cluster_data = features[clusters == i]
            result.append({
                'id': i,
                'center': {
                    'price': self.kmeans.cluster_centers_[i][0],
                    'area': self.kmeans.cluster_centers_[i][1]
                },
                'size': len(cluster_data),
                'avg_price': cluster_data['price'].mean(),
                'avg_area': cluster_data['area'].mean()
            })
        
        return result

    def generate_price_heatmap(self):
        """Генерация тепловой карты цен"""
        # Подготовка данных для тепловой карты
        heatmap_data = []
        for _, row in self.sale_df.iterrows():
            if pd.isna(row['price']) or pd.isna(row['area']):
                continue
            
            price_per_m2 = row['price'] / row['area']
            heatmap_data.append({
                'lat': row.get('lat'),  # Предполагаем, что есть координаты
                'lon': row.get('lon'),
                'weight': price_per_m2
            })
        
        return heatmap_data

    def find_investment_opportunities(self):
        """Поиск инвестиционных возможностей"""
        opportunities = []
        
        for _, sale in self.sale_df.iterrows():
            if pd.isna(sale['area']) or pd.isna(sale['price']):
                continue
            
            # Находим похожие объекты в аренде
            similar_rent = self.rent_df[
                (self.rent_df['area'].between(sale['area'] * 0.9, sale['area'] * 1.1)) &
                (self.rent_df['address'].str.contains(sale['address'].split(',')[0], na=False))
            ]
            
            if len(similar_rent) > 0:
                avg_monthly_rent = similar_rent['price'].mean()
                if avg_monthly_rent > 0:
                    # Расчет срока окупаемости в годах
                    payback_period = sale['price'] / (avg_monthly_rent * 12)
                    
                    opportunities.append({
                        'address': sale['address'],
                        'area': sale['area'],
                        'sale_price': sale['price'],
                        'avg_monthly_rent': avg_monthly_rent,
                        'payback_years': payback_period,
                        'roi': (avg_monthly_rent * 12 / sale['price']) * 100  # ROI в процентах
                    })
        
        # Сортируем по сроку окупаемости
        opportunities.sort(key=lambda x: x['payback_years'])
        return opportunities[:10]  # Возвращаем топ-10 возможностей

    def generate_basic_stats(self):
        """Генерация базовой статистики по обоим наборам данных"""
        stats = {
            'rent': {
                'price': self.rent_df['price'].describe().to_dict(),
                'area': self.rent_df['area'].describe().to_dict()
            },
            'sale': {
                'price': self.sale_df['price'].describe().to_dict(),
                'area': self.sale_df['area'].describe().to_dict()
            }
        }
        return stats

    def generate_plots(self, output_dir='plots'):
        """Генерация графиков для анализа данных"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Создаем графики для данных аренды
        self._create_analysis_plots(self.rent_df, 'rent', output_dir)
        
        # Создаем графики для данных продажи
        self._create_analysis_plots(self.sale_df, 'sale', output_dir)

    def _create_analysis_plots(self, df, data_type, output_dir):
        plt.figure(figsize=(15, 10))

        # 1. Распределение цен
        plt.subplot(2, 2, 1)
        sns.histplot(data=df, x='price', bins=30)
        plt.title(f'Распределение цен ({data_type})')
        plt.xlabel('Цена')
        plt.ylabel('Количество')

        # 2. Распределение площадей
        plt.subplot(2, 2, 2)
        sns.histplot(data=df, x='area', bins=30)
        plt.title(f'Распределение площадей ({data_type})')
        plt.xlabel('Площадь (м²)')
        plt.ylabel('Количество')

        # 3. Зависимость цены от площади
        plt.subplot(2, 2, 3)
        sns.scatterplot(data=df, x='area', y='price')
        plt.title(f'Зависимость цены от площади ({data_type})')
        plt.xlabel('Площадь (м²)')
        plt.ylabel('Цена')

        # 4. Динамика цен по времени
        plt.subplot(2, 2, 4)
        df.groupby('date')['price'].mean().plot()
        plt.title(f'Динамика средних цен по времени ({data_type})')
        plt.xlabel('Дата')
        plt.ylabel('Средняя цена')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'{data_type}_analysis.png'))
        plt.close() 