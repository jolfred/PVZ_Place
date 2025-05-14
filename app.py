from flask import Flask, render_template, jsonify, request, flash, redirect, url_for
from data_cleaning.pipeline import DataCleaner
from analysis import RealEstateAnalyzer
from config import Config
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go
import json
import os
from datetime import datetime

app = Flask(__name__)
app.config.from_object(Config)

# Проверка наличия API ключа
if not app.config['YANDEX_MAPS_API_KEY']:
    raise ValueError('YANDEX_MAPS_API_KEY не найден в переменных окружения. Создайте файл .env и добавьте в него ключ.')

print(f"Яндекс API ключ загружен успешно: {app.config['YANDEX_MAPS_API_KEY'][:8]}...")

# Пути к данным
DATA_DIR = os.path.join('data_cleaning', 'data', 'processed')
RENT_DATA_PATH = os.path.join(DATA_DIR, 'rent_data.csv')
SALE_DATA_PATH = os.path.join(DATA_DIR, 'sale_data.csv')

# Инициализация анализатора
analyzer = RealEstateAnalyzer()

# Загружаем фейковые данные (потом замените на свои)
pvz_data = [
    {"id": 1, "city": "Москва", "lat": 55.751244, "lon": 37.618423, "address": "ул. Тверская, 1", "competitors": 3,
     "population": 5000, "traffic": "high"},
    {"id": 2, "city": "Москва", "lat": 55.753215, "lon": 37.622504, "address": "ул. Охотный ряд, 2", "competitors": 1,
     "population": 8000, "traffic": "very_high"},
]

# Список городов Татарстана
CITIES = {
    'kazan': 'Казань',
    'naberezhnye-chelny': 'Набережные Челны',
    'nizhnekamsk': 'Нижнекамск',
    'almetyevsk': 'Альметьевск',
    'zelenodolsk': 'Зеленодольск'
}

# Главная страница (карта + список точек)
@app.route("/")
def index():
    return render_template("index.html", points=pvz_data, cities=CITIES)


# API: Поиск точек по городу
@app.route("/api/search")
def search():
    city = request.args.get("city", "Москва")
    filtered_points = [p for p in pvz_data if p["city"] == city]
    return jsonify(filtered_points)


# API: Анализ конкретной точки
@app.route("/api/analyze/<int:id>")
def analyze(id):
    point = next((p for p in pvz_data if p["id"] == id), None)
    if not point:
        return jsonify({"error": "Точка не найдена"}), 404

    # Простейший рейтинг (0-100)
    score = (point["population"] / 10000 * 40) + (60 if point["traffic"] == "very_high" else 30)
    point["score"] = min(100, int(score))

    return jsonify(point)


@app.route("/analysis")
def analysis():
    return render_template("analysis.html")


@app.route("/api/market_overview")
def market_overview():
    try:
        stats = analyzer.get_market_statistics()
        return jsonify(stats)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/property_clusters")
def property_clusters():
    try:
        clusters = analyzer.get_property_clusters()
        return jsonify(clusters)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/investment_opportunities")
def investment_opportunities():
    try:
        opportunities = analyzer.find_investment_opportunities()
        return jsonify(opportunities)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/property_analysis", methods=['POST'])
def analyze_property():
    try:
        property_data = request.json
        analysis_result = analyzer.analyze_single_property(property_data)
        return jsonify(analysis_result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/price_heatmap")
def price_heatmap():
    try:
        heatmap_data = analyzer.generate_price_heatmap()
        return jsonify(heatmap_data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/city/<city_slug>')
def city_view(city_slug):
    if city_slug not in CITIES:
        flash('Город не найден', 'error')
        return redirect(url_for('index'))
    
    city_name = CITIES[city_slug]
    return render_template('city.html', 
                         city_name=city_name, 
                         city_slug=city_slug)


@app.route('/api/city/<city_slug>/stats')
def city_stats(city_slug):
    try:
        if city_slug not in CITIES:
            return jsonify({'error': 'Город не найден'}), 404
            
        city_name = CITIES[city_slug]
        stats = analyzer.get_city_statistics(city_name)
        return jsonify(stats)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/city/<city_slug>/clusters')
def city_clusters(city_slug):
    try:
        if city_slug not in CITIES:
            return jsonify({'error': 'Город не найден'}), 404
            
        city_name = CITIES[city_slug]
        # Фильтруем кластеры для конкретного города
        all_clusters = analyzer.get_property_clusters()
        city_clusters = [c for c in all_clusters if c['size'] > 0]  # Добавим фильтрацию по городу позже
        return jsonify(city_clusters)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/city/<city_slug>/opportunities')
def city_opportunities(city_slug):
    try:
        if city_slug not in CITIES:
            return jsonify({'error': 'Город не найден'}), 404
            
        city_name = CITIES[city_slug]
        # Получаем все возможности и фильтруем по городу
        all_opportunities = analyzer.find_investment_opportunities()
        city_opportunities = [
            opp for opp in all_opportunities 
            if city_name.lower() in opp['address'].lower()
        ]
        return jsonify(city_opportunities)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/city/<city_slug>/heatmap')
def city_heatmap(city_slug):
    try:
        if city_slug not in CITIES:
            return jsonify({'error': 'Город не найден'}), 404
            
        city_name = CITIES[city_slug]
        # Получаем данные тепловой карты и фильтруем по городу
        all_heatmap_data = analyzer.generate_price_heatmap()
        city_heatmap_data = [
            data for data in all_heatmap_data 
            if data.get('lat') is not None and data.get('lon') is not None
        ]
        return jsonify(city_heatmap_data)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# Запуск сервера
if __name__ == "__main__":
    app.run(debug=True, host='localhost', port=5000)