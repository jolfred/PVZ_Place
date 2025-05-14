// Глобальные переменные
let map;
let citySlug;
let cityName;
let clusters = [];
let opportunities = [];

// Функция инициализации страницы города
async function initCityPage(slug, name) {
    citySlug = slug;
    cityName = name;
    
    // Инициализируем карту
    await initMap();
    
    // Загружаем данные
    loadCityData();
}

// Инициализация карты
async function initMap() {
    // Ждем загрузки API
    await ymaps3.ready;
    
    const {YMap, YMapDefaultSchemeLayer} = ymaps3;

    // Координаты центров городов
    const cityCoordinates = {
        'kazan': [49.073303, 55.7955015],
        'naberezhnye-chelny': [52.3959886, 55.7437185],
        'nizhnekamsk': [51.8045272, 55.6366164],
        'almetyevsk': [52.2972564, 54.9013662],
        'zelenodolsk': [48.5010882, 55.8466751]
    };

    // Создаем карту
    map = new YMap(
        document.getElementById('cityMap'),
        {
            location: {
                center: cityCoordinates[citySlug] || cityCoordinates['kazan'],
                zoom: 12
            }
        }
    );

    // Добавляем слой схемы
    map.addChild(new YMapDefaultSchemeLayer());
}

// Загрузка данных города
async function loadCityData() {
    try {
        // Загружаем все данные параллельно
        const [statsResponse, clustersResponse, heatmapResponse] = await Promise.all([
            fetch(`/api/city/${citySlug}/stats`),
            fetch(`/api/city/${citySlug}/clusters`),
            fetch(`/api/city/${citySlug}/heatmap`)
        ]);

        const [statsData, clustersData, heatmapData] = await Promise.all([
            statsResponse.json(),
            clustersResponse.json(),
            heatmapResponse.json()
        ]);

        // Обновляем интерфейс
        updateStatistics(statsData);
        updateClusters(clustersData);
        updateHeatmap(heatmapData);
    } catch (error) {
        console.error('Error loading city data:', error);
        alert('Ошибка при загрузке данных города');
    }
}

// Обновление статистики
function updateStatistics(data) {
    document.getElementById('avgPrice').textContent = data.avg_price ? formatPrice(data.avg_price) : 'Нет данных';
    document.getElementById('pricePerM2').textContent = data.price_per_m2 ? formatPrice(data.price_per_m2) : 'Нет данных';
    document.getElementById('totalObjects').textContent = data.total_objects || 0;
    document.getElementById('priceTrend').textContent = data.price_trend ? `${data.price_trend}%` : 'Нет данных';

    // Создаем график цен только если есть данные
    if (data.price_chart && data.price_chart.data && data.price_chart.layout) {
        Plotly.newPlot('priceChart', data.price_chart.data, data.price_chart.layout);
    } else {
        document.getElementById('priceChart').innerHTML = '<div class="text-center text-muted py-5">Недостаточно данных для построения графика</div>';
    }
}

// Обновление кластеров
async function updateClusters(clustersData) {
    clusters = clustersData;
    
    if (!map) return;

    const {YMapDefaultSchemeLayer, YMapPolygon} = ymaps3;
    
    // Очищаем карту
    map.removeAllChildren();
    map.addChild(new YMapDefaultSchemeLayer());
    
    // Добавляем кластеры на карту
    clusters.forEach(cluster => {
        const color = cluster.color || '#3498db';
        
        // Создаем полигон кластера
        const polygon = new YMapPolygon({
            geometry: {
                coordinates: cluster.coordinates,
                type: 'Polygon'
            },
            properties: {
                hint: `Кластер ${cluster.name}`,
                balloonContent: `
                    <strong>${cluster.name}</strong><br>
                    Средняя цена: ${formatPrice(cluster.avg_price)}<br>
                    Объектов: ${cluster.objects_count}
                `
            },
            style: {
                fillColor: color,
                fillOpacity: 0.3,
                strokeColor: color,
                strokeWidth: 2
            }
        });
        
        map.addChild(polygon);
    });
    
    // Обновляем список кластеров
    updateClustersList();
}

// Обновление списка кластеров
function updateClustersList() {
    const container = document.getElementById('clustersList');
    container.innerHTML = clusters.map(cluster => `
        <div class="cluster-card p-3 mb-3 border rounded" style="border-left: 4px solid ${cluster.color || '#3498db'} !important">
            <h6 class="mb-2">${cluster.name}</h6>
            <div class="small text-muted mb-2">
                <div>Средняя цена: ${formatPrice(cluster.avg_price)}</div>
                <div>Объектов: ${cluster.objects_count}</div>
            </div>
            <button class="btn btn-sm btn-outline-primary" onclick="loadClusterOpportunities(${cluster.id})">
                Показать предложения
            </button>
        </div>
    `).join('');
}

// Загрузка предложений кластера
async function loadClusterOpportunities(clusterId) {
    try {
        const response = await fetch(`/api/city/${citySlug}/opportunities?cluster_id=${clusterId}`);
        const data = await response.json();
        updateOpportunitiesList(data);
    } catch (error) {
        console.error('Error loading opportunities:', error);
        alert('Ошибка при загрузке предложений');
    }
}

// Обновление списка предложений
function updateOpportunitiesList(opportunitiesData) {
    opportunities = opportunitiesData;
    const container = document.getElementById('opportunitiesList');
    
    container.innerHTML = opportunities.map(opp => `
        <div class="opportunity-card p-3 mb-3 border rounded">
            <div class="d-flex justify-content-between align-items-start">
                <div>
                    <h6 class="mb-1">${opp.address}</h6>
                    <div class="small text-muted mb-2">
                        <div>Площадь: ${opp.area} м²</div>
                        <div>Цена: ${formatPrice(opp.price)}</div>
                        <div>Цена за м²: ${formatPrice(opp.price_per_m2)}</div>
                    </div>
                </div>
                <div class="text-end">
                    <div class="badge bg-success mb-2">-${Math.round(opp.discount)}%</div>
                </div>
            </div>
            <button class="btn btn-sm btn-outline-primary w-100" onclick="showPropertyDetails(${opp.id})">
                Подробнее
            </button>
        </div>
    `).join('');
}

// Отображение деталей объекта
async function showPropertyDetails(propertyId) {
    try {
        const response = await fetch(`/api/property/${propertyId}`);
        const property = await response.json();
        
        const modal = new bootstrap.Modal(document.getElementById('propertyModal'));
        
        document.getElementById('modalTitle').textContent = property.address;
        document.getElementById('modalBody').innerHTML = `
            <div class="row">
                <div class="col-md-6">
                    <h6>Характеристики объекта</h6>
                    <ul class="list-unstyled">
                        <li><strong>Площадь:</strong> ${property.area} м²</li>
                        <li><strong>Цена:</strong> ${formatPrice(property.price)}</li>
                        <li><strong>Цена за м²:</strong> ${formatPrice(property.price_per_m2)}</li>
                        <li><strong>Комнат:</strong> ${property.rooms}</li>
                        <li><strong>Этаж:</strong> ${property.floor}</li>
                    </ul>
                </div>
                <div class="col-md-6">
                    <h6>Инвестиционный потенциал</h6>
                    <div class="progress mb-2">
                        <div class="progress-bar bg-success" style="width: ${property.investment_potential}%"></div>
                    </div>
                    <p class="small text-muted">
                        Потенциал роста: ${property.investment_potential}%
                    </p>
                    <div class="mt-3">
                        <h6>Преимущества</h6>
                        <ul class="small">
                            ${property.advantages.map(adv => `<li>${adv}</li>`).join('')}
                        </ul>
                    </div>
                </div>
            </div>
        `;
        
        modal.show();
    } catch (error) {
        console.error('Error loading property details:', error);
        alert('Ошибка при загрузке данных объекта');
    }
}

// Форматирование цены
function formatPrice(price) {
    if (!price || isNaN(price)) return 'Нет данных';
    return new Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'RUB',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(price);
} 