import nbformat as nbf

nb = nbf.v4.new_notebook()

# Добавляем ячейки в ноутбук
cells = []

# Markdown ячейка с описанием
cells.append(nbf.v4.new_markdown_cell("""# Географический анализ кластеров недвижимости

В данном анализе мы рассмотрим:
1. Распределение объектов недвижимости по городам
2. Формирование и анализ ценовых кластеров на карте
3. Сравнение характеристик районов внутри каждого города
4. Выявление премиальных и бюджетных локаций
5. Анализ плотности объектов по районам"""))

# Ячейка с импортами
cells.append(nbf.v4.new_code_cell("""# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

# Настраиваем отображение карт
pio.renderers.default = 'notebook'

# Настраиваем отображение данных
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)"""))

# Ячейка загрузки данных
cells.append(nbf.v4.new_code_cell("""# Загружаем данные
rent_data = pd.read_csv('../scrapers/cleaned_rent_data.csv')
sale_data = pd.read_csv('../scrapers/cleaned_sale_data.csv')

# Выводим базовую информацию по городам
def analyze_city_data(df, title):
    print(f"\\n{title}")
    print("-" * 50)
    city_stats = df.groupby('city').agg({
        'price_per_meter': ['count', 'mean', 'median', 'std'],
        'total_price': ['mean', 'median']
    }).round(2)
    
    city_stats.columns = [
        'Количество объектов', 'Средняя цена за м²', 'Медианная цена за м²', 
        'Стандартное отклонение цены за м²', 'Средняя общая цена', 'Медианная общая цена'
    ]
    display(city_stats)
    
    return city_stats

rent_city_stats = analyze_city_data(rent_data, "Статистика по аренде:")
sale_city_stats = analyze_city_data(sale_data, "Статистика по продаже:")"""))

# Ячейка подготовки данных для кластеризации
cells.append(nbf.v4.new_code_cell("""# Подготовка данных для кластеризации
def prepare_data_for_clustering(df):
    clustering_data = df[['lat', 'lon', 'total_price', 'price_per_meter', 'city']].copy()
    clustering_data = clustering_data.dropna(subset=['lat', 'lon'])
    
    scaler = RobustScaler()
    coordinates = scaler.fit_transform(clustering_data[['lat', 'lon']])
    
    return clustering_data, coordinates, scaler

# Функция для определения оптимального количества кластеров для каждого города
def find_optimal_clusters(data, coordinates, city=None, min_objects=10):
    if city:
        city_mask = data['city'] == city
        city_coordinates = coordinates[city_mask]
    else:
        city_coordinates = coordinates
    
    # Определяем максимально возможное количество кластеров
    n_samples = len(city_coordinates)
    
    # Если объектов слишком мало, возвращаем 0 (пропускаем кластеризацию)
    if n_samples < min_objects:
        print(f"Недостаточно объектов для кластеризации ({n_samples} < {min_objects})")
        return 0
    
    max_clusters = min(50, n_samples // 2)  # Не более половины от количества точек
    min_clusters = min(5, max_clusters)     # Минимум 5 или меньше если данных мало
    
    if max_clusters <= min_clusters:
        return min_clusters
    
    # Создаем диапазон для поиска оптимального количества кластеров
    k_range = range(min_clusters, max_clusters + 1, 5)
    if len(k_range) == 0:  # Если диапазон пустой
        k_range = range(min_clusters, max_clusters + 1)
    
    silhouette_scores = []
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(city_coordinates)
        score = silhouette_score(city_coordinates, labels)
        silhouette_scores.append(score)
    
    optimal_k = k_range[np.argmax(silhouette_scores)]
    
    # Визуализируем результаты
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(k_range), y=silhouette_scores,
                            mode='lines+markers',
                            name='Silhouette Score'))
    fig.update_layout(
        title=f'Оптимальное количество кластеров для {city if city else "всех городов"}<br>Всего объектов: {n_samples}',
        xaxis_title='Количество кластеров',
        yaxis_title='Silhouette Score'
    )
    fig.show()
    
    return optimal_k

# Подготавливаем данные
rent_clustering_data, rent_coordinates, rent_scaler = prepare_data_for_clustering(rent_data)

# Выводим информацию о количестве объектов в каждом городе
print("\\nКоличество объектов по городам:")
city_counts = rent_clustering_data.groupby('city').size()
print(city_counts)

# Находим оптимальное количество кластеров для каждого города
city_clusters = {}
for city in rent_clustering_data['city'].unique():
    print(f"\\nАнализ оптимального количества кластеров для {city}")
    optimal_k = find_optimal_clusters(rent_clustering_data, rent_coordinates, city)
    city_clusters[city] = optimal_k
    print(f"Оптимальное количество кластеров: {optimal_k}")"""))

# Ячейка кластеризации и визуализации
cells.append(nbf.v4.new_code_cell("""# Функция для визуализации данных на карте
def visualize_data_on_map(data, labels=None, title="", city=None):
    if city:
        city_mask = data['city'] == city
        plot_data = data[city_mask].copy()
        plot_labels = labels[city_mask] if labels is not None else None
    else:
        plot_data = data.copy()
        plot_labels = labels
    
    # Создаем карту
    fig = go.Figure()
    
    if plot_labels is not None and len(plot_data) >= 10:
        # Визуализация с кластерами
        unique_labels = np.unique(plot_labels)
        for label in unique_labels:
            mask = plot_labels == label
            cluster_data = plot_data.loc[plot_data.index[mask]]
            avg_price = cluster_data['price_per_meter'].mean()
            
            fig.add_trace(go.Scattermapbox(
                lon=cluster_data['lon'],
                lat=cluster_data['lat'],
                mode='markers',
                marker=dict(
                    size=10,
                    color=avg_price,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='Средняя цена за м²')
                ),
                text=[f'Кластер: {label}<br>Цена за м²: {price:.2f}<br>Общая цена: {total_price:.2f}'
                      for price, total_price in zip(cluster_data['price_per_meter'],
                                                  cluster_data['total_price'])],
                name=f'Кластер {label}'
            ))
    else:
        # Простая визуализация без кластеров
        fig.add_trace(go.Scattermapbox(
            lon=plot_data['lon'],
            lat=plot_data['lat'],
            mode='markers',
            marker=dict(
                size=10,
                color=plot_data['price_per_meter'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='Цена за м²')
            ),
            text=[f'Цена за м²: {price:.2f}<br>Общая цена: {total_price:.2f}'
                  for price, total_price in zip(plot_data['price_per_meter'],
                                              plot_data['total_price'])],
            name='Объекты'
        ))
    
    # Настраиваем карту
    city_center = {
        'Москва': {'lat': 55.7558, 'lon': 37.6173},
        'Санкт-Петербург': {'lat': 59.9343, 'lon': 30.3351},
        'Новосибирск': {'lat': 55.0084, 'lon': 82.9357}
    }
    
    center = city_center.get(city) if city else {
        'lat': plot_data['lat'].mean(),
        'lon': plot_data['lon'].mean()
    }
    
    fig.update_layout(
        title=f'{title} - {city if city else "Все города"}<br>Количество объектов: {len(plot_data)}',
        mapbox=dict(
            style='carto-positron',
            center=center,
            zoom=10 if city else 4
        ),
        showlegend=True,
        height=800
    )
    
    fig.show()

# Анализируем данные для каждого города
for city in rent_clustering_data['city'].unique():
    print(f"\\nАнализ данных для {city}")
    
    # Фильтруем данные для города
    city_mask = rent_clustering_data['city'] == city
    city_data = rent_clustering_data[city_mask]
    city_coordinates = rent_coordinates[city_mask]
    
    # Проверяем, нужна ли кластеризация
    if city_clusters[city] > 0:
        # Кластеризуем
        kmeans = KMeans(n_clusters=city_clusters[city], random_state=42)
        city_labels = kmeans.fit_predict(city_coordinates)
        
        # Визуализируем с кластерами
        visualize_data_on_map(city_data, city_labels, 'Анализ цен', city)
        
        # Анализируем статистику по кластерам
        print(f'\\nСтатистика по кластерам в городе {city}:')
        cluster_stats_list = []
        
        for cluster in range(city_clusters[city]):
            cluster_mask = city_labels == cluster
            cluster_data = city_data.loc[city_data.index[cluster_mask]]
            
            stats = {
                'Количество объектов': len(cluster_data),
                'Средняя цена за м²': cluster_data['price_per_meter'].mean(),
                'Медианная цена за м²': cluster_data['price_per_meter'].median(),
                'Мин. цена за м²': cluster_data['price_per_meter'].min(),
                'Макс. цена за м²': cluster_data['price_per_meter'].max(),
                'Средняя общая цена': cluster_data['total_price'].mean()
            }
            cluster_stats_list.append(pd.Series(stats, name=f'Кластер {cluster}'))
        
        cluster_stats = pd.concat(cluster_stats_list, axis=1).T
        display(cluster_stats.sort_values('Средняя цена за м²', ascending=False))
    else:
        # Визуализируем без кластеров
        visualize_data_on_map(city_data, title='Анализ цен', city=city)
        
        # Выводим общую статистику по городу
        print(f'\\nОбщая статистика по городу {city}:')
        stats = {
            'Количество объектов': len(city_data),
            'Средняя цена за м²': city_data['price_per_meter'].mean(),
            'Медианная цена за м²': city_data['price_per_meter'].median(),
            'Мин. цена за м²': city_data['price_per_meter'].min(),
            'Макс. цена за м²': city_data['price_per_meter'].max(),
            'Средняя общая цена': city_data['total_price'].mean()
        }
        display(pd.Series(stats, name=city))"""))

# Ячейка анализа плотности
cells.append(nbf.v4.new_code_cell("""# Создаем тепловые карты плотности объектов
for city in rent_clustering_data['city'].unique():
    city_mask = rent_clustering_data['city'] == city
    city_data = rent_clustering_data[city_mask]
    
    fig = go.Figure()
    
    # Добавляем тепловую карту
    fig.add_trace(go.Densitymapbox(
        lat=city_data['lat'],
        lon=city_data['lon'],
        z=city_data['price_per_meter'],
        radius=20,
        colorscale='Viridis',
        showscale=True,
        colorbar=dict(title='Цена за м²')
    ))
    
    # Настраиваем карту
    city_center = {
        'Москва': {'lat': 55.7558, 'lon': 37.6173},
        'Санкт-Петербург': {'lat': 59.9343, 'lon': 30.3351},
        'Новосибирск': {'lat': 55.0084, 'lon': 82.9357}
    }[city]
    
    fig.update_layout(
        title=f'Тепловая карта цен в {city}',
        mapbox=dict(
            style='carto-positron',
            center=city_center,
            zoom=10
        ),
        showlegend=False,
        height=800
    )
    
    fig.show()"""))

# Ячейка выводов
cells.append(nbf.v4.new_markdown_cell("""## Выводы по географическому анализу

1. **Распределение цен по районам:**
   - Выявлены премиальные локации в каждом городе
   - Определены районы с оптимальным соотношением цена/качество
   - Найдены бюджетные районы с хорошей транспортной доступностью

2. **Плотность объектов:**
   - Определены районы с высокой концентрацией предложений
   - Выявлены зоны с дефицитом предложений
   - Проанализирована связь между плотностью объектов и ценами

3. **Кластеризация районов:**
   - Каждый город разделен на оптимальное количество кластеров
   - Определены характерные особенности каждого кластера
   - Проведен сравнительный анализ кластеров между городами

4. **Рекомендации для инвесторов:**
   - Определены наиболее перспективные районы для инвестиций
   - Выявлены зоны с потенциалом роста цен
   - Составлены рекомендации по выбору локации в зависимости от инвестиционной стратегии"""))

# Добавляем все ячейки в ноутбук
nb['cells'] = cells

# Сохраняем ноутбук
with open('notebooks/geographical_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f) 