import nbformat as nbf

nb = nbf.v4.new_notebook()

# Добавляем ячейки в ноутбук
cells = []

# Markdown ячейка с описанием
cells.append(nbf.v4.new_markdown_cell("""# Расширенный анализ инвестиционных объектов недвижимости

В данном анализе мы проведем:
1. Предварительный анализ данных и их очистку
2. Углубленный анализ ценовых трендов
3. Сегментацию рынка с использованием продвинутых методов кластеризации
4. Прогнозирование потенциальной доходности
5. Оценку рисков инвестирования
6. Формирование инвестиционных рекомендаций"""))

# Ячейка с импортами
cells.append(nbf.v4.new_code_cell("""# Импортируем необходимые библиотеки
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import silhouette_score
from sklearn.ensemble import IsolationForest
from scipy import stats
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Настраиваем отображение
%matplotlib inline
plt.style.use('seaborn')
pd.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.2f' % x)"""))

# Ячейка загрузки и первичного анализа данных
cells.append(nbf.v4.new_code_cell("""# Загружаем данные
rent_data = pd.read_csv('../scrapers/cleaned_rent_data.csv')
sale_data = pd.read_csv('../scrapers/cleaned_sale_data.csv')

def analyze_dataset(df, name):
    print(f'\\nАнализ датасета: {name}')
    print('-' * 50)
    print(f'Количество записей: {len(df)}')
    print('\\nОписательная статистика числовых признаков:')
    display(df.describe())
    print('\\nПропущенные значения:')
    display(df.isnull().sum())
    
    # Определяем выбросы в ценах
    Q1 = df['price_per_meter'].quantile(0.25)
    Q3 = df['price_per_meter'].quantile(0.75)
    IQR = Q3 - Q1
    outliers = df[(df['price_per_meter'] < (Q1 - 1.5 * IQR)) | 
                  (df['price_per_meter'] > (Q3 + 1.5 * IQR))]
    print(f'\\nКоличество выбросов в ценах: {len(outliers)}')
    
    return outliers

# Анализируем оба датасета
rent_outliers = analyze_dataset(rent_data, 'Аренда')
sale_outliers = analyze_dataset(sale_data, 'Продажа')"""))

# Ячейка визуализации распределения цен
cells.append(nbf.v4.new_code_cell("""# Создаем интерактивную визуализацию распределения цен
fig = make_subplots(rows=2, cols=2,
                    subplot_titles=('Распределение цен аренды',
                                  'Распределение цен продажи',
                                  'Box-plot цен аренды по городам',
                                  'Box-plot цен продажи по городам'))

# Гистограммы распределения цен
fig.add_trace(
    go.Histogram(x=rent_data['price_per_meter'], name='Аренда',
                 nbinsx=50, histnorm='probability'),
    row=1, col=1
)

fig.add_trace(
    go.Histogram(x=sale_data['price_per_meter'], name='Продажа',
                 nbinsx=50, histnorm='probability'),
    row=1, col=2
)

# Box-plots по городам
fig.add_trace(
    go.Box(x=rent_data['city'], y=rent_data['price_per_meter'],
           name='Аренда'),
    row=2, col=1
)

fig.add_trace(
    go.Box(x=sale_data['city'], y=sale_data['price_per_meter'],
           name='Продажа'),
    row=2, col=2
)

fig.update_layout(height=800, showlegend=False)
fig.show()"""))

# Ячейка анализа корреляций и зависимостей
cells.append(nbf.v4.new_code_cell("""# Анализ корреляций между характеристиками объектов
def analyze_correlations(df, title):
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    corr_matrix = df[numeric_cols].corr()
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title(f'Корреляционная матрица - {title}')
    plt.show()
    
    # Находим сильные корреляции
    strong_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if abs(corr_matrix.iloc[i, j]) > 0.5:
                strong_corr.append({
                    'var1': corr_matrix.columns[i],
                    'var2': corr_matrix.columns[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    
    if strong_corr:
        print(f'\\nСильные корреляции ({title}):')
        for corr in strong_corr:
            print(f"{corr['var1']} - {corr['var2']}: {corr['correlation']:.2f}")

analyze_correlations(rent_data, 'Аренда')
analyze_correlations(sale_data, 'Продажа')"""))

# Ячейка продвинутой кластеризации
cells.append(nbf.v4.new_code_cell("""# Продвинутая кластеризация с использованием различных алгоритмов
def advanced_clustering(data, coordinates):
    results = {}
    
    # KMeans
    kmeans = KMeans(n_clusters=50, random_state=42)
    kmeans_labels = kmeans.fit_predict(coordinates)
    results['kmeans'] = {
        'labels': kmeans_labels,
        'silhouette': silhouette_score(coordinates, kmeans_labels)
    }
    
    # DBSCAN
    dbscan = DBSCAN(eps=0.3, min_samples=5)
    dbscan_labels = dbscan.fit_predict(coordinates)
    if len(np.unique(dbscan_labels)) > 1:  # Проверяем, что есть больше одного кластера
        results['dbscan'] = {
            'labels': dbscan_labels,
            'silhouette': silhouette_score(coordinates, dbscan_labels)
        }
    
    # Визуализация результатов
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('KMeans кластеризация',
                                      'DBSCAN кластеризация'))
    
    # KMeans plot
    fig.add_trace(
        go.Scatter(x=data['lon'], y=data['lat'],
                  mode='markers',
                  marker=dict(color=kmeans_labels,
                            colorscale='Viridis'),
                  name='KMeans'),
        row=1, col=1
    )
    
    # DBSCAN plot
    fig.add_trace(
        go.Scatter(x=data['lon'], y=data['lat'],
                  mode='markers',
                  marker=dict(color=dbscan_labels,
                            colorscale='Viridis'),
                  name='DBSCAN'),
        row=1, col=2
    )
    
    fig.update_layout(height=500, title_text="Сравнение методов кластеризации")
    fig.show()
    
    return results

# Подготовка данных
def prepare_data_for_clustering(df):
    clustering_data = df[['lat', 'lon', 'total_price', 'price_per_meter']].copy()
    clustering_data = clustering_data.dropna(subset=['lat', 'lon'])
    
    scaler = RobustScaler()  # Используем RobustScaler для лучшей устойчивости к выбросам
    coordinates = scaler.fit_transform(clustering_data[['lat', 'lon']])
    
    return clustering_data, coordinates, scaler

# Выполняем кластеризацию
rent_clustering_data, rent_coordinates, rent_scaler = prepare_data_for_clustering(rent_data)
rent_clustering_results = advanced_clustering(rent_clustering_data, rent_coordinates)

print('\\nРезультаты кластеризации:')
for method, results in rent_clustering_results.items():
    print(f'{method}: silhouette score = {results["silhouette"]:.3f}')"""))

# Ячейка анализа инвестиционных возможностей
cells.append(nbf.v4.new_code_cell("""# Расширенный анализ инвестиционных возможностей
def analyze_investment_opportunities(sale_data, rent_data, kmeans_labels):
    # Рассчитываем метрики для каждого кластера
    cluster_metrics = pd.DataFrame()
    
    for cluster in np.unique(kmeans_labels):
        cluster_rent = rent_data[kmeans_labels == cluster]
        
        metrics = {
            'avg_rent': cluster_rent['total_price'].mean(),
            'median_rent': cluster_rent['total_price'].median(),
            'rent_volatility': cluster_rent['total_price'].std() / cluster_rent['total_price'].mean(),
            'num_properties': len(cluster_rent)
        }
        
        cluster_metrics = cluster_metrics.append(
            pd.Series(metrics, name=cluster)
        )
    
    # Находим перспективные объекты
    def calculate_roi(row, cluster_metrics):
        cluster = kmeans_labels[row.name]
        if cluster in cluster_metrics.index:
            annual_rent = cluster_metrics.loc[cluster, 'median_rent'] * 12
            roi = (annual_rent - row['total_price'] * 0.05) / row['total_price'] * 100
            return roi
        return np.nan
    
    sale_data['roi'] = sale_data.apply(
        lambda x: calculate_roi(x, cluster_metrics), axis=1
    )
    
    # Оцениваем риски
    sale_data['risk_score'] = sale_data.apply(
        lambda x: cluster_metrics.loc[kmeans_labels[x.name], 'rent_volatility']
        if kmeans_labels[x.name] in cluster_metrics.index else np.nan,
        axis=1
    )
    
    # Визуализация результатов
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=('ROI vs Риск',
                                      'Распределение ROI',
                                      'Распределение рисков',
                                      'Топ кластеры по ROI'))
    
    # ROI vs Риск
    fig.add_trace(
        go.Scatter(x=sale_data['risk_score'],
                  y=sale_data['roi'],
                  mode='markers',
                  marker=dict(color=sale_data['total_price'],
                            colorscale='Viridis',
                            showscale=True),
                  name='Объекты'),
        row=1, col=1
    )
    
    # Распределение ROI
    fig.add_trace(
        go.Histogram(x=sale_data['roi'],
                    name='ROI'),
        row=1, col=2
    )
    
    # Распределение рисков
    fig.add_trace(
        go.Histogram(x=sale_data['risk_score'],
                    name='Риски'),
        row=2, col=1
    )
    
    # Топ кластеры по ROI
    top_clusters = sale_data.groupby(kmeans_labels)['roi'].mean().nlargest(10)
    fig.add_trace(
        go.Bar(x=top_clusters.index,
               y=top_clusters.values,
               name='Топ кластеры'),
        row=2, col=2
    )
    
    fig.update_layout(height=800, showlegend=False)
    fig.show()
    
    return sale_data[['total_price', 'roi', 'risk_score']]

# Анализируем инвестиционные возможности
investment_analysis = analyze_investment_opportunities(
    sale_clustering_data,
    rent_clustering_data,
    rent_clustering_results['kmeans']['labels']
)

# Выводим топ-10 наиболее привлекательных объектов
print('\\nТоп-10 объектов для инвестирования:')
display(investment_analysis.sort_values('roi', ascending=False).head(10))"""))

# Ячейка рекомендаций по инвестированию
cells.append(nbf.v4.new_markdown_cell("""## Рекомендации по инвестированию

На основе проведенного анализа можно сделать следующие выводы:

1. **Наиболее привлекательные районы:**
   - Районы с высоким ROI и низким риском
   - Районы с стабильным потоком арендаторов
   - Районы с потенциалом роста цен

2. **Критерии выбора объектов:**
   - Соотношение цена/качество
   - Потенциальный доход от аренды
   - Риски и волатильность цен

3. **Стратегии инвестирования:**
   - Долгосрочная аренда
   - Краткосрочная аренда
   - Перепродажа после ремонта

4. **Факторы риска:**
   - Волатильность цен в районе
   - Конкуренция
   - Состояние объекта"""))

# Добавляем все ячейки в ноутбук
nb['cells'] = cells

# Сохраняем ноутбук
with open('notebooks/advanced_real_estate_analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)