from flask import Flask, render_template, jsonify, request

import pandas as pd

app = Flask(__name__)

# Загружаем фейковые данные (потом замените на свои)
pvz_data = [
    {"id": 1, "city": "Москва", "lat": 55.751244, "lon": 37.618423, "address": "ул. Тверская, 1", "competitors": 3,
     "population": 5000, "traffic": "high"},
    {"id": 2, "city": "Москва", "lat": 55.753215, "lon": 37.622504, "address": "ул. Охотный ряд, 2", "competitors": 1,
     "population": 8000, "traffic": "very_high"},
]


# Главная страница (карта + список точек)
@app.route("/")
def index():
    return render_template("index.html", points=pvz_data)


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


# Запуск сервера
if __name__ == "__main__":
    app.run(host='localhost', port=5000)
    # app.run(debug=True)