// Функция для форматирования цен
function formatPrice(price) {
    return new Intl.NumberFormat('ru-RU', {
        style: 'currency',
        currency: 'RUB',
        minimumFractionDigits: 0,
        maximumFractionDigits: 0
    }).format(price);
}

// Функция для отображения тепловой карты
function displayHeatmap(data) {
    const heatmapData = JSON.parse(data);
    Plotly.newPlot('priceHeatmap', heatmapData.data, heatmapData.layout);
}

// Функция для отображения статистики рынка
function displayMarketStats(data) {
    const stats = data.stats;
    const plotData = JSON.parse(data.plot);
    
    // Обновляем статистику
    document.getElementById('avgPrice').textContent = formatPrice(stats.avg_price);
    document.getElementById('pricePerM2').textContent = formatPrice(stats.price_per_m2);
    document.getElementById('totalObjects').textContent = stats.total_objects;
    document.getElementById('priceTrend').textContent = `${stats.price_trend}%`;
    
    // Отображаем график
    Plotly.newPlot('marketStats', plotData.data, plotData.layout);
}

// Функция для отображения инвестиционных возможностей
function displayOpportunities(opportunities) {
    const container = document.getElementById('investmentOpportunities');
    container.innerHTML = opportunities.map(opp => `
        <div class="card mb-3 opportunity-card">
            <div class="card-body">
                <div class="row">
                    <div class="col-md-8">
                        <h5 class="card-title">${opp.district}</h5>
                        <p class="card-text">
                            <strong>Площадь:</strong> ${opp.area} м²<br>
                            <strong>Цена:</strong> ${formatPrice(opp.price)}<br>
                            <strong>Цена за м²:</strong> ${formatPrice(opp.price_per_m2)}
                        </p>
                    </div>
                    <div class="col-md-4 text-end">
                        <div class="stats-number text-primary">${Math.round(opp.price_diff / opp.price_per_m2 * 100)}%</div>
                        <div class="stats-label">Потенциал роста</div>
                        <button class="btn btn-outline-primary mt-2" onclick="showPropertyDetails(${JSON.stringify(opp)})">
                            Подробнее
                        </button>
                    </div>
                </div>
            </div>
        </div>
    `).join('');
}

// Функция для отображения деталей объекта
function showPropertyDetails(property) {
    // Создаем модальное окно с деталями
    const modal = new bootstrap.Modal(document.getElementById('propertyModal'));
    
    // Заполняем данные
    document.getElementById('modalTitle').textContent = `${property.district} - ${property.area} м²`;
    document.getElementById('modalBody').innerHTML = `
        <div class="row">
            <div class="col-md-6">
                <h6>Основные характеристики</h6>
                <ul class="list-unstyled">
                    <li><strong>Цена:</strong> ${formatPrice(property.price)}</li>
                    <li><strong>Цена за м²:</strong> ${formatPrice(property.price_per_m2)}</li>
                    <li><strong>Этаж:</strong> ${property.floor}</li>
                    <li><strong>Комнат:</strong> ${property.rooms}</li>
                </ul>
            </div>
            <div class="col-md-6">
                <h6>Инвестиционный потенциал</h6>
                <div class="progress mb-2">
                    <div class="progress-bar bg-primary" role="progressbar" 
                         style="width: ${Math.round(property.price_diff / property.price_per_m2 * 100)}%">
                    </div>
                </div>
                <p class="small text-muted">
                    Объект недооценен на ${Math.round(property.price_diff / property.price_per_m2 * 100)}% 
                    относительно среднерыночной цены в районе
                </p>
            </div>
        </div>
    `;
    
    modal.show();
}

// Обработчик отправки формы анализа
document.getElementById('analysisForm').addEventListener('submit', async function(e) {
    e.preventDefault();
    
    const formData = {
        district: document.getElementById('district').value,
        propertyType: document.getElementById('propertyType').value,
        budget: parseFloat(document.getElementById('budget').value)
    };
    
    try {
        // Загружаем все данные параллельно
        const [heatmapResponse, statsResponse, opportunitiesResponse] = await Promise.all([
            fetch('/api/price_heatmap'),
            fetch('/api/market_overview'),
            fetch('/api/investment_opportunities')
        ]);
        
        const [heatmapData, statsData, opportunitiesData] = await Promise.all([
            heatmapResponse.json(),
            statsResponse.json(),
            opportunitiesResponse.json()
        ]);
        
        // Отображаем все данные
        displayHeatmap(heatmapData);
        displayMarketStats(statsData);
        displayOpportunities(opportunitiesData);
        
    } catch (error) {
        console.error('Error fetching data:', error);
        alert('Произошла ошибка при загрузке данных. Пожалуйста, попробуйте позже.');
    }
}); 