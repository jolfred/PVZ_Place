import logging
from datetime import datetime
import os
from .pipeline import DataCleaner

def clean_sale_data():
    """
    Функция для очистки данных о продаже
    """
    # Создаем экземпляр очистки данных для продажи
    cleaner = DataCleaner('scrapers/merged_sale_data.csv', 'sale')
    
    # Выполняем очистку
    cleaned_df = cleaner.clean_data()
    
    # Сохраняем результаты
    cleaner.save_results('scrapers/cleaned_sale_data.csv')
    
    # Генерируем и выводим отчет
    report = cleaner.generate_report()
    
    logging.info("\n=== Отчет по очистке данных продажи ===")
    logging.info(f"Всего записей: {report['total_records']}")
    
    logging.info("\nРаспределение по городам:")
    for city, count in report['cities'].items():
        logging.info(f"{city}: {count}")
    
    logging.info("\nСтатистика по ценам продажи:")
    for stat, value in report['total_price_stats'].items():
        logging.info(f"{stat}: {value:,.2f}")
    
    logging.info("\nСтатистика по ценам продажи за м²:")
    for stat, value in report['price_per_meter_stats'].items():
        logging.info(f"{stat}: {value:.2f}")
    
    logging.info("\nСтатистика по площадям:")
    for stat, value in report['area_stats'].items():
        logging.info(f"{stat}: {value:.2f}")
    
    # Выводим информацию об экстремальных ценах
    cleaner.print_extreme_prices()

if __name__ == "__main__":
    # Создаем директорию для логов, если она не существует
    os.makedirs('logs', exist_ok=True)

    # Настройка логирования
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/sale_cleaning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    
    clean_sale_data() 