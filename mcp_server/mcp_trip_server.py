"""
需求：实现基于MCP的行程管家服务器，提供租车、旅游团、保险的查询及预订功能

架构说明：
    本服务器是 SmartVoyage 系统中的另一个 MCP 服务器，运行在 8003 端口。
    与 mcp_ticket_server.py 保持一致的架构模式，但采用了混合数据源：

    - 租车查询：从 MySQL car_rentals 表真实查询
    - 保险查询：从 MySQL insurances 表真实查询
    - 旅游团查询：从 Milvus 向量数据库进行语义检索（RAG 模式）
    - 预订功能：模拟返回（待后续完善）

    旅游团 RAG 系统说明：
    - 使用 Milvus 本地向量数据库存储旅游团信息
    - 旅游团数据在初始化时通过 Qwen Embedding API 生成 1024 维向量
    - 用户查询时，将查询文本也生成向量，通过 COSINE 相似度匹配最相关的旅游团
    - 优势：用户可以用自然语言描述需求（如"想看雪山的地方"），系统自动匹配最合适的旅游团
"""

# ==================== 导入依赖 ====================
import json  # JSON 处理
from datetime import date, datetime, timedelta  # 时间处理
from decimal import Decimal  # 高精度小数类型
import requests  # HTTP 请求，用于调用 Qwen Embedding API

import pymysql  # MySQL 数据库驱动
from pymilvus import connections, Collection  # Milvus 向量数据库客户端
from fastmcp import FastMCP
# from mcp.server.fastmcp import FastMCP  # FastMCP 框架，快速创建 MCP 服务器

from config import Config  # 项目配置
from create_logger import logger  # 日志模块
from utils.format import DateEncoder, default_encoder  # 日期格式化工具

conf = Config()  # 全局配置实例


# ==================== Milvus + Embedding 配置 ====================
MILVUS_HOST = "localhost"
MILVUS_PORT = 19530
COLLECTION_NAME = "tour_groups"
EMBEDDING_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/embeddings"
EMBEDDING_DIM = 1024


def get_embedding(text: str) -> list:
    """
    调用 Qwen Embedding API 生成文本向量（1024 维）

    参数：
        text (str): 需要生成向量的文本

    返回值：
        list: 1024 维浮点向量
    """
    headers = {
        "Authorization": f"Bearer {conf.api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "text-embedding-v3",
        "input": [text],
        "dimensions": EMBEDDING_DIM
    }
    response = requests.post(EMBEDDING_URL, headers=headers, json=payload, timeout=30)
    result = response.json()
    return result["data"][0]["embedding"]


def search_tour_groups_in_milvus(query_text: str, city: str = None, limit: int = 5) -> list:
    """
    在 Milvus 中进行语义搜索，找到最匹配的旅游团

    工作流程：
    1. 将用户查询文本通过 Qwen API 转换为 1024 维向量
    2. 在 Milvus 中用 COSINE 相似度搜索最相近的旅游团
    3. 返回匹配结果

    参数：
        query_text (str): 用户查询文本（如"想看雪山的地方"）
        city (str, optional): 城市过滤条件，如果有则先过滤再搜索
        limit (int): 返回结果数量，默认 5

    返回值：
        list: 匹配的旅游团列表
    """
    # 连接 Milvus
    connections.connect(alias="default", host=MILVUS_HOST, port=MILVUS_PORT)
    collection = Collection(COLLECTION_NAME)
    collection.load()

    # 如果指定了城市，先做城市过滤
    if city:
        # 先用城市过滤，再在结果中做向量搜索
        # Milvus 不支持先过滤再向量搜索的简单方式，我们用 expr 过滤
        results = collection.search(
            data=[get_embedding(query_text)],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["tour_id", "tour_name", "city", "days", "price",
                           "total_seats", "remaining_seats", "agency", "rating",
                           "departure_dates", "highlights"],
            expr=f'city == "{city}"' if city else None
        )
    else:
        results = collection.search(
            data=[get_embedding(query_text)],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=limit,
            output_fields=["tour_id", "tour_name", "city", "days", "price",
                           "total_seats", "remaining_seats", "agency", "rating",
                           "departure_dates", "highlights"]
        )

    # 将搜索结果转换为字典列表
    tour_list = []
    for hits in results:
        for hit in hits:
            tour_list.append({
                "tour_id": hit.entity.get("tour_id"),
                "tour_name": hit.entity.get("tour_name"),
                "city": hit.entity.get("city"),
                "days": hit.entity.get("days"),
                "price": hit.entity.get("price"),
                "total_seats": hit.entity.get("total_seats"),
                "remaining_seats": hit.entity.get("remaining_seats"),
                "agency": hit.entity.get("agency"),
                "rating": round(hit.entity.get("rating"), 1),
                "departure_dates": hit.entity.get("departure_dates"),
                "highlights": hit.entity.get("highlights"),
                "similarity": round(hit.distance, 4),
            })

    return tour_list


# ==================== 行程服务类 ====================
class TripService:
    """
    行程查询与预订服务类

    混合数据源架构：
    - 租车、保险：从 MySQL 数据库真实查询
    - 旅游团：从 Milvus 向量数据库语义检索（RAG）
    """

    def __init__(self):
        # FastMCP 会在线程池中并发执行同步工具，MySQL 连接不能跨线程共享；
        # 连接长时间空闲也会被服务端断开，因此每次查询独立建立连接。
        pass

    def _connect(self):
        return pymysql.connect(
            host=conf.host,
            user=conf.user,
            password=conf.password,
            database=conf.database,
            autocommit=True
        )

    def _execute_query(self, sql: str, params: list = None) -> str:
        """
        执行 SQL 查询，返回 JSON 格式的结果

        参数：
            sql (str): SQL 查询语句，使用 %s 占位符
            params (list): SQL 参数列表

        返回值：
            str: JSON 字符串
        """
        conn = None
        cursor = None
        try:
            conn = self._connect()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params or [])
            results = cursor.fetchall()

            # 处理特殊类型（日期、Decimal 等）
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = default_encoder(value)

            if results:
                response = {"status": "success", "data": results}
            else:
                response = {"status": "no_data", "message": "未找到行程数据，请确认查询条件。"}

            return json.dumps(response, cls=DateEncoder, ensure_ascii=False)

        except Exception as e:
            logger.error(f"行程查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    # ========== 租车查询（MySQL） ==========

    def query_car_rental(self, pickup_city: str, return_city: str, date: str, car_type: str = None) -> str:
        """
        查询租车信息（从 MySQL car_rentals 表查询）
        """
        logger.info(f"查询租车: {pickup_city} -> {return_city}, {date}, {car_type}")
        sql = ("SELECT id, company, pickup_city, return_city, pickup_date, car_type, "
               "car_model, price_per_day, total_available, transmission, seats, deposit "
               "FROM car_rentals "
               "WHERE pickup_city = %s AND return_city = %s AND pickup_date = %s")
        params = [pickup_city, return_city, date]
        if car_type:
            sql += " AND car_type = %s"
            params.append(car_type)
        return self._execute_query(sql, params)

    # ========== 旅游团查询（Milvus RAG） ==========

    def query_tour_group(self, query_text: str, city: str = None) -> str:
        """
        查询旅游团信息（从 Milvus 向量数据库进行语义检索）

        参数：
            query_text (str): 用户查询描述，如"想看雪山的地方"、"适合亲子游的短途旅行"
            city (str, optional): 城市过滤条件

        返回值：
            str: JSON 字符串，包含匹配的旅游团列表
        """
        logger.info(f"查询旅游团: query='{query_text}', city='{city}'")
        try:
            tour_list = search_tour_groups_in_milvus(query_text, city, limit=5)

            if tour_list:
                return json.dumps({"status": "success", "data": tour_list}, ensure_ascii=False)
            else:
                return json.dumps({
                    "status": "no_data",
                    "message": f"未找到匹配的旅游团。{'请确认城市名称，' if city else ''}或尝试其他搜索条件。"
                }, ensure_ascii=False)

        except Exception as e:
            logger.error(f"旅游团 RAG 查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": f"旅游团查询出错：{str(e)}"}, ensure_ascii=False)

    # ========== 保险查询（MySQL） ==========

    def query_insurance(self, insurance_type: str = None) -> str:
        """
        查询旅行保险产品信息（从 MySQL insurances 表查询）
        """
        logger.info(f"查询保险: {insurance_type}")
        sql = ("SELECT id, insurance_type, name, company, coverage, price, "
               "duration_days, max_coverage, medical_coverage, baggage_coverage, flight_delay "
               "FROM insurances")
        params = []
        if insurance_type:
            sql += " WHERE insurance_type = %s"
            params.append(insurance_type)
        return self._execute_query(sql, params)

    # ========== 预订方法 ==========

    def order_car_rental(self, date: str, car_type: str, number: int) -> str:
        """预订租车"""
        logger.info(f"预订租车: {date}, {car_type}, {number}辆")
        return f"恭喜，租车预订成功！日期：{date}，车型：{car_type}，数量：{number}辆。"

    def order_tour_group(self, date: str, tour_name: str, number: int) -> str:
        """报名旅游团"""
        logger.info(f"报名旅游团: {date}, {tour_name}, {number}人")
        return f"恭喜，旅游团报名成功！团名：{tour_name}，日期：{date}，人数：{number}人。"

    def order_insurance(self, insurance_type: str, date: str, number: int) -> str:
        """购买旅行保险"""
        logger.info(f"购买保险: {insurance_type}, {date}, {number}份")
        return f"恭喜，旅行保险购买成功！类型：{insurance_type}，日期：{date}，份数：{number}份。"


# ==================== 创建 MCP 服务器 ====================
def create_trip_mcp_server():
    """
    创建并启动行程管家 MCP 服务器
    """
    trip_mcp = FastMCP(
        name="TripTools",
        instructions="行程管家工具。租车和保险通过 MySQL 查询，旅游团通过 Milvus 语义检索。支持查询和预订。",
        # log_level="ERROR",
        # host="127.0.0.1", port=8003
    )

    service = TripService()

    # ========== 注册查询工具 ==========

    @trip_mcp.tool(
        name="query_car_rental",
        description="查询租车信息，参数：pickup_city(取车城市), return_city(还车城市), date(日期，格式YYYY-MM-DD), car_type(车型类型，可选：经济型/SUV/豪华型/MPV)"
    )
    def query_car_rental(pickup_city: str, return_city: str, date: str, car_type: str = None) -> str:
        return service.query_car_rental(pickup_city, return_city, date, car_type)

    @trip_mcp.tool(
        name="query_tour_group",
        description="查询旅游团信息（语义搜索），参数：query_text(查询描述，如'想看雪山的地方')，city(城市过滤，可选)"
    )
    def query_tour_group(query_text: str, city: str = None) -> str:
        return service.query_tour_group(query_text, city)

    @trip_mcp.tool(
        name="query_insurance",
        description="查询旅行保险产品，参数：insurance_type(保险类型，可选：综合型/意外型/医疗型/境外型)"
    )
    def query_insurance(insurance_type: str = None) -> str:
        return service.query_insurance(insurance_type)

    # ========== 注册预订工具 ==========

    @trip_mcp.tool(
        name="order_car_rental",
        description="根据日期、车型、数量预订租车服务"
    )
    def order_car_rental(date: str, car_type: str, number: int) -> str:
        logger.info(f"正在预订租车: {date}, {car_type}, {number}辆")
        return service.order_car_rental(date, car_type, number)

    @trip_mcp.tool(
        name="order_tour_group",
        description="根据日期、团名、人数报名旅游团"
    )
    def order_tour_group(date: str, tour_name: str, number: int) -> str:
        logger.info(f"正在报名旅游团: {date}, {tour_name}, {number}人")
        return service.order_tour_group(date, tour_name, number)

    @trip_mcp.tool(
        name="order_insurance",
        description="根据保险类型、日期、份数购买旅行保险"
    )
    def order_insurance(insurance_type: str, date: str, number: int) -> str:
        logger.info(f"正在购买保险: {insurance_type}, {date}, {number}份")
        return service.order_insurance(insurance_type, date, number)

    # 打印服务器信息
    logger.info("=== 行程管家MCP服务器信息 ===")
    logger.info(f"名称: {trip_mcp.name}")
    logger.info(f"描述: {trip_mcp.instructions}")

    try:
        print("服务器已启动，请访问 http://127.0.0.1:8003/mcp")
        trip_mcp.run(transport="http", host="0.0.0.0", port=8003)
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == '__main__':
    create_trip_mcp_server()
