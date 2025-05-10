import osmnx as ox
import pandas as pd
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut


# Функция для геокодирования адреса
def geocode_address(address):
    geolocator = Nominatim(user_agent="geoapiExercises")
    try:
        location = geolocator.geocode(address)
        return (location.latitude, location.longitude) if location else None
    except GeocoderTimedOut:
        print("Время ожидания геокодирования истекло.")
        return None


# Функция для получения информации о трафике
def get_traffic_data(lat, lon):
    # Получаем граф улиц вокруг заданных координат
    graph = ox.graph_from_point((lat, lon), dist=1000, network_type='drive')
    # Получаем данные о дорожной сети
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)

    return edges[['name', 'length', 'highway']]


# Основная функция
def main():
    address = input("Введите адрес или координаты (широта,долгота): ")

    if ',' in address:
        try:
            lat, lon = map(float, address.split(','))
        except ValueError:
            print("Некорректный формат координат.")
            return
    else:
        coords = geocode_address(address)
        if coords:
            lat, lon = coords
        else:
            print("Не удалось найти координаты для данного адреса.")
            return

    traffic_data = get_traffic_data(lat, lon)

    if not traffic_data.empty:
        print("\nДанные о дорожной сети:")
        print(traffic_data)

        # Сохранение данных в CSV
        traffic_data.to_csv('traffic_data.csv', index=False, encoding='utf-8')
        print("Данные сохранены в файл 'traffic_data.csv'.")
    else:
        print("Не удалось получить данные о дорожной сети.")


if __name__ == "__main__":
    main()
