# app/agent/tools.py
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from datetime import datetime, timedelta
from typing import Optional


# ============================================================
# База данных товаров (в реальном проекте — обращение к БД)
# ============================================================

PRODUCTS = {
    1: {"name": "Ноутбук Lenovo", "price": 75000, "in_stock": True, "category": "laptop"},
    2: {"name": "Ноутбук Acer",   "price": 78000, "in_stock": True, "category": "laptop"},
    3: {"name": "Ноутбук HP",     "price": 82000, "in_stock": True, "category": "laptop"},
    4: {"name": "Смартфон Samsung","price": 40000, "in_stock": True, "category": "phone"},
    5: {"name": "Наушники Sony",  "price": 15000, "in_stock": True, "category": "audio"},
    6: {"name": "Ноутбук Asus",   "price": 65000, "in_stock": False,"category": "laptop"},
}

# Словарь для хранения корзин по сессиям
SESSION_CARTS = {}


# ============================================================
# Схемы входных данных
# ============================================================

class ProductSearchInfo(BaseModel):
    query: str = Field(
        description="Поисковый запрос для поиска по названию товара (например: 'ноутбук')"
    )
    max_price: int = Field(
        default=9999999999,
        description="Максимальная цена товара. Если не указана — 9999999999"
    )
    min_price: int = Field(
        default=0,
        description="Минимальная цена товара. Если не указана — 0"
    )
    in_stock_only: bool = Field(
        default=False,
        description="Если True — возвращать только товары в наличии"
    )

class CartOperationInfo(BaseModel):
    product_id: int = Field(
        description="ID товара для операций с корзиной"
    )
    session_id: str = Field(
        description="ID сессии пользователя"
    )

class CartSummaryInfo(BaseModel):
    session_id: str = Field(
        description="ID сессии пользователя"
    )
    promo_code: Optional[str] = Field(
        default=None,
        description="Промокод для скидки"
    )


# ============================================================
# Инструменты
# ============================================================

@tool(args_schema=ProductSearchInfo)
def search_products(
    query: str,
    max_price: int = 9999999999,
    min_price: int = 0,
    in_stock_only: bool = False
) -> list:
    """Поиск товаров по названию, цене и наличию на складе."""
    keywords = query.lower().split()
    results = []

    for pid, data in PRODUCTS.items():
        name = data["name"].lower()
        price = data["price"]
        in_stock = data["in_stock"]

        matches_query = any(word in name for word in keywords)
        matches_price = min_price <= price <= max_price
        matches_stock = not in_stock_only or in_stock

        if matches_query and matches_price and matches_stock:
            results.append({
                "id": pid,
                "name": data["name"],
                "price": price,
                "in_stock": in_stock,
            })

    return results


@tool
def check_delivery_date(city: str) -> str:
    """Возвращает ожидаемую дату доставки в указанный город."""
    days = 1 if city.lower() == "минск" else 3
    delivery_date = datetime.now() + timedelta(days=days)
    return f"Доставка в город {city} ожидается {delivery_date.strftime('%d.%m.%Y')}"


@tool(args_schema=CartOperationInfo)
def add_to_cart(product_id: int, session_id: str) -> str:
    """Добавляет товар в корзину по его ID для указанной сессии."""
    if product_id not in PRODUCTS:
        return f"Товар с ID {product_id} не найден."

    product = PRODUCTS[product_id]

    if not product["in_stock"]:
        return f"Товар «{product['name']}» сейчас не в наличии."

    # Получаем корзину для сессии или создаем новую
    if session_id not in SESSION_CARTS:
        SESSION_CARTS[session_id] = []
    
    SESSION_CARTS[session_id].append(product_id)
    # print(f"Товар «{product['name']}» добавлен в корзину. Session ID: {session_id}") 
    return f"Товар «{product['name']}» добавлен в корзину."


@tool
def get_product_info(product_id: int) -> str:
    """Возвращает подробную информацию о товаре по его ID."""
    if product_id not in PRODUCTS:
        return f"Товар с ID {product_id} не найден."

    p = PRODUCTS[product_id]
    status = "в наличии" if p["in_stock"] else "нет в наличии"
    return (
        f"Название: {p['name']}\n"
        f"Цена: {p['price']:,} руб.\n"
        f"Статус: {status}\n"
        f"Категория: {p['category']}"
    )


@tool(args_schema=CartSummaryInfo)
def get_cart_summary(session_id: str, promo_code: Optional[str] = None) -> str:
    """Возвращает состав корзины и итоговую сумму для указанной сессии. Принимает промокод для скидки."""
    if session_id not in SESSION_CARTS or not SESSION_CARTS[session_id]:
        return "Корзина пуста."

    lines = []
    total = 0

    for pid in SESSION_CARTS[session_id]:
        product = PRODUCTS[pid]
        lines.append(f"- {product['name']}: {product['price']:,} руб.")
        total += product["price"]

    discount_info = ""
    if promo_code and promo_code.lower() == "promo":
        total = int(total * 0.9)
        discount_info = " (скидка 10% по промокоду PROMO)"

    lines.append(f"\nИтого{discount_info}: {total:,} руб.")
    return "\n".join(lines)


# ============================================================
# Фабричная функция — возвращает список инструментов
# ============================================================

def get_tools(retriever=None) -> list:
    """
    Возвращает список всех инструментов агента.
    retriever — опционально, если передан, добавляется инструмент поиска по базе знаний.
    """
    tools = [
        search_products,
        check_delivery_date,
        add_to_cart,
        get_product_info,
        get_cart_summary,
    ]

    if retriever is not None:
        @tool
        def search_knowledge_base(query: str) -> str:
            """Поиск информации о магазине в базе знаний: доставка, гарантия, оплата, акции.
            Также используй для получения подробного описания товара."""
            docs = retriever.invoke(query)
            if not docs:
                return "Информация по запросу не найдена."
            return "\n\n---\n\n".join(doc.page_content for doc in docs)

        tools.append(search_knowledge_base)

    return tools