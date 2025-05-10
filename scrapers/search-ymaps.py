import requests
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

# Функция для получения ближайших точек притяжения
def get_nearby_places(lat, lon, radius):
    query = f"""
    [out:json];
    (
      node["amenity"="pharmacy"](around:{radius},{lat},{lon});
      node["shop"="supermarket"](around:{radius},{lat},{lon});
      node["amenity"="bus_station"](around:{radius},{lat},{lon});
      node["amenity"="metro_station"](around:{radius},{lat},{lon});
      node["shop"="mall"](around:{radius},{lat},{lon});
      node["shop"="pickup_point"](around:{radius},{lat},{lon}); // ПВЗ
      node["name"~"Ozon|Wildberries|Яндекс.Маркет"](around:{radius},{lat},{lon}); // ПВЗ Ozon, Wildberries, Яндекс.Маркет
    );
    out body;
    """

    try:
        response = requests.get("http://overpass-api.de/api/interpreter", params={'data': query})
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Ошибка при запросе к Overpass API: {e}")
        return None

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

    all_places = []

    for radius in [300, 600]:
        print(f"\nДанные для радиуса {radius} метров:")
        places_data = get_nearby_places(lat, lon, radius)
        if places_data and 'elements' in places_data:
            for place in places_data['elements']:
                all_places.append({
                    'Name': place.get('tags', {}).get('name', 'Неизвестно'),
                    'Type': place.get('tags', {}).get('amenity', place.get('tags', {}).get('shop', 'Неизвестно')),
                    'Radius': radius,
                    'Latitude': place['lat'],
                    'Longitude': place['lon']
                })
                print(
                    f"- {place.get('tags', {}).get('name', 'Неизвестно')} ({place.get('tags', {}).get('amenity', place.get('tags', {}).get('shop', 'Неизвестно'))})")
        else:
            print("Не удалось получить данные о местах.")

    if all_places:
        df = pd.DataFrame(all_places)
        df.to_csv('nearby_places.csv', index=False, encoding='utf-8')
        print("\nДанные сохранены в файл 'nearby_places.csv'.")

if __name__ == "__main__":
    main()
