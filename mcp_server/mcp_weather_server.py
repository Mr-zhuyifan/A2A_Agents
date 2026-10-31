"""
需求：实现基于MCP的天气查询服务器，提供参数化的天气查询工具

架构说明：
    MCP Server 提供带明确参数的 tool 函数（city、start_date、end_date），
    内部根据配置项 `conf.weather_source` 决定数据来源：
    - "database"：从 MySQL 数据库查询（需要 spider_weather.py 定时更新）
    - "api"：直接从和风天气 API 实时获取（无需数据库）

    与票务 MCP（mcp_ticket_server.py）保持一致的架构模式：
    - MCP Server 提供带明确参数的 tool 函数
    - A2A server 使用 LangChain Agent + MCP Tools，让 LLM 自动从用户输入中提取参数
"""

# ==================== 导入依赖 ====================
import gzip  # gzip 解压，和风 API 返回 gzip 压缩数据
import json  # JSON 处理，用于序列化查询结果
import os  # 读取环境变量
from datetime import date, datetime, timedelta  # 时间处理
from decimal import Decimal  # 高精度小数类型

import pymysql  # MySQL 数据库驱动
import requests  # HTTP 请求库，用于调用和风天气 API

# from mcp.server.fastmcp import FastMCP  # FastMCP 框架，快速创建 MCP 服务器
from fastmcp import FastMCP
from config import Config  # 项目配置（导入时会自动加载项目根目录 .env）
from create_logger import logger  # 日志模块
from utils.format import DateEncoder, default_encoder  # 日期格式化工具

conf = Config()  # 全局配置实例


# ==================== 和风天气 API 配置 ====================
# 和风天气 API 密钥（从环境变量 / .env 读取，参考 .env.example）
HEFENG_API_KEY = os.getenv("QWEATHER_MCP_API_KEY", "").strip()
# 和风天气 API 基础地址（不同账号的专属域名不同，从环境变量读取）
HEFENG_BASE_URL = os.getenv(
    "QWEATHER_MCP_BASE_URL", "https://devapi.qweather.com/v7/weather/30d"
)

# 城市名称到和风城市代码的映射
# 和风 API 使用城市代码而不是城市名称来查询天气
CITY_CODES = {
    "北京": "101010100",
    "上海": "101020100",
    "杭州": "101280101",
    "南京": "101190101",
    "广州": "101280101",
    "深圳": "101280601",
}


def fetch_weather_from_api(city: str, start_date: str, end_date: str = None) -> str:
    """
    直接从和风天气 API 获取天气数据

    当配置中 weather_source = "api" 时使用。
    调用和风天气 API 获取指定城市和日期范围的天气预报数据。

    参数：
        city (str): 城市名称，如 "北京"
        start_date (str): 开始日期，格式 YYYY-MM-DD
        end_date (str, optional): 结束日期，格式 YYYY-MM-DD

    返回值：
        str: JSON 字符串，格式与数据库查询保持一致：
            - {"status": "success", "data": [...]}  # 查询成功
            - {"status": "no_data", "message": "..."}  # 无数据
            - {"status": "error", "message": "..."}  # 执行出错
    """
    # 检查城市代码是否存在
    location = CITY_CODES.get(city)
    if not location:
        return json.dumps({
            "status": "error",
            "message": f"不支持的城市：{city}。目前支持的城市有：{', '.join(CITY_CODES.keys())}"
        }, ensure_ascii=False)

    # 调用和风天气 API
    if not HEFENG_API_KEY:
        return json.dumps({
            "status": "error",
            "message": "未配置和风天气 Key，请在项目根目录 .env 中设置 QWEATHER_MCP_API_KEY（参考 .env.example）"
        }, ensure_ascii=False)

    headers = {
        "X-QW-Api-Key": HEFENG_API_KEY,
        "Accept-Encoding": "gzip"
    }
    url = f"{HEFENG_BASE_URL}?location={location}"

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        # 和风 API 有时返回 gzip 压缩数据，有时返回纯 JSON 文本
        # 策略：先尝试判断是否为 JSON，如果是则直接解析；否则尝试 gzip 解压
        raw_content = response.content
        if raw_content.startswith(b'{') or raw_content.startswith(b'['):
            # 看起来是 JSON 文本，直接解析
            data = response.text
        elif response.headers.get('Content-Encoding') == 'gzip':
            # 是 gzip 压缩数据，解压后解析
            data = gzip.decompress(raw_content).decode('utf-8')
        else:
            # 默认直接解析
            data = response.text

        api_result = json.loads(data)

        # 检查 API 返回状态码
        if api_result.get("code") != "200":
            return json.dumps({
                "status": "error",
                "message": f"和风天气 API 返回错误码：{api_result.get('code')}"
            }, ensure_ascii=False)

        # 从 API 返回的原始数据中提取并格式化
        daily_data = api_result.get("daily", [])
        if not daily_data:
            return json.dumps({
                "status": "no_data",
                "message": "和风天气 API 未返回预报数据。"
            }, ensure_ascii=False)

        # 将 API 返回的数据格式化为与数据库查询一致的格式
        formatted_data = []
        for day in daily_data:
            fx_date = day.get("fxDate", "")

            # 如果指定了日期范围，则过滤
            if start_date and fx_date < start_date:
                continue
            if end_date and fx_date > end_date:
                break

            formatted_data.append({
                "city": city,
                "fx_date": fx_date,
                "temp_max": int(day.get("tempMax", 0)),
                "temp_min": int(day.get("tempMin", 0)),
                "text_day": day.get("textDay", ""),
                "text_night": day.get("textNight", ""),
                "humidity": int(day.get("humidity", 0)),
                "wind_dir_day": day.get("windDirDay", ""),
                "wind_scale_day": day.get("windScaleDay", ""),
                "wind_speed_day": int(day.get("windSpeedDay", 0)),
                "precip": float(day.get("precip", 0.0)),
                "uv_index": int(day.get("uvIndex", 0)),
                "pressure": int(day.get("pressure", 0)),
                "vis": int(day.get("vis", 0)),
                "cloud": int(day.get("cloud", 0)),
                "sunrise": day.get("sunrise", ""),
                "sunset": day.get("sunset", ""),
                "moonrise": day.get("moonrise", ""),
                "moonset": day.get("moonset", ""),
                "moon_phase": day.get("moonPhase", ""),
                "icon_day": day.get("iconDay", ""),
                "icon_night": day.get("iconNight", ""),
                "wind360_day": int(day.get("wind360Day", 0)),
                "wind360_night": int(day.get("wind360Night", 0)),
                "wind_dir_night": day.get("windDirNight", ""),
                "wind_scale_night": day.get("windScaleNight", ""),
                "wind_speed_night": int(day.get("windSpeedNight", 0)),
                "update_time": api_result.get("updateTime", ""),
            })

        if not formatted_data:
            return json.dumps({
                "status": "no_data",
                "message": f"未找到 {city} 在 {start_date} 到 {end_date or start_date} 之间的天气数据。"
            }, ensure_ascii=False)

        return json.dumps({"status": "success", "data": formatted_data}, cls=DateEncoder, ensure_ascii=False)

    except requests.RequestException as e:
        logger.error(f"请求 {city} 和风天气 API 失败: {e}")
        return json.dumps({"status": "error", "message": f"和风天气 API 请求失败：{str(e)}"}, ensure_ascii=False)
    except json.JSONDecodeError as e:
        logger.error(f"{city} 和风天气 JSON 解析错误: {e}")
        return json.dumps({"status": "error", "message": f"和风天气 JSON 解析失败：{str(e)}"}, ensure_ascii=False)
    except Exception as e:
        logger.error(f"和风天气 API 查询出错: {str(e)}")
        return json.dumps({"status": "error", "message": f"和风天气 API 查询出错：{str(e)}"}, ensure_ascii=False)


# ==================== 天气服务类（数据库模式） ====================
class WeatherService:
    """
    天气查询服务类（数据库模式），封装数据库操作逻辑

    当配置中 weather_source = "database" 时使用。
    负责：
    1. 维护数据库连接
    2. 提供参数化的查询方法（由参数自动拼接 SQL）
    3. 格式化查询结果为 JSON 返回
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

    def execute_query(self, sql: str, params: list = None) -> str:
        """
        执行 SQL 查询，返回 JSON 格式的结果

        使用参数化查询（params 参数）防止 SQL 注入。

        参数：
            sql (str): SQL 查询语句，使用 %s 占位符
            params (list): SQL 参数列表，对应 %s 占位符

        返回值：
            str: JSON 字符串，格式为以下之一：
                - {"status": "success", "data": [...]}  # 查询成功
                - {"status": "no_data", "message": "..."}  # 无数据
                - {"status": "error", "message": "..."}  # 执行出错
        """
        conn = None
        cursor = None
        try:
            conn = self._connect()
            # 使用 DictCursor，返回的结果每一行是一个字典（字段名为 key）
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params or [])
            results = cursor.fetchall()

            # 处理特殊类型（日期、Decimal 等），转换为 JSON 可序列化的格式
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = default_encoder(value)

            # 根据查询结果构造不同的 JSON 响应
            if results:
                response = {"status": "success", "data": results}
            else:
                response = {"status": "no_data", "message": "未找到天气数据，请确认城市和日期。"}

            return json.dumps(response, cls=DateEncoder, ensure_ascii=False)

        except Exception as e:
            logger.error(f"天气查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    def query_weather(self, city: str, start_date: str, end_date: str = None) -> str:
        """
        参数化天气查询（数据库模式）—— 根据城市和日期范围查询天气数据

        接收结构化的参数，由 MCP Server 内部拼接 SQL，保证安全性。

        参数：
            city (str): 城市名称，如 "北京"
            start_date (str): 开始日期，格式 YYYY-MM-DD，如 "2025-07-30"
            end_date (str, optional): 结束日期，格式 YYYY-MM-DD
                                     如果为 None，则只查询 start_date 当天
                                     如果不为 None，则查询 start_date 到 end_date 的范围

        返回值：
            str: JSON 字符串，包含天气数据
        """
        # 基础查询字段：选择用户最常关心的天气信息
        sql = (
            "SELECT city, fx_date, temp_max, temp_min, text_day, text_night, "
            "humidity, wind_dir_day, precip "
            "FROM weather_data "
            "WHERE city = %s"
        )
        params = [city]

        if end_date and end_date != start_date:
            # 有结束日期且与开始日期不同 → 日期范围查询
            sql += " AND fx_date BETWEEN %s AND %s ORDER BY fx_date"
            params.extend([start_date, end_date])
        else:
            # 单天查询
            sql += " AND fx_date = %s"
            params.append(start_date)

        logger.info(f"天气查询参数: city={city}, start_date={start_date}, end_date={end_date}")
        logger.info(f"生成的SQL: {sql}, 参数: {params}")

        return self.execute_query(sql, params)


# ==================== 创建 MCP 服务器 ====================
def create_weather_mcp_server():
    """
    创建并启动天气查询 MCP 服务器

    根据配置项 conf.weather_source 决定使用哪种数据源：
    - "database"：使用 WeatherService 从数据库查询
    - "api"：直接从和风天气 API 获取
    """
    # 创建 FastMCP 实例（MCP 服务器）
    weather_mcp = FastMCP(
        name="WeatherTools",
        instructions="天气查询工具，支持参数化查询。数据来源可配置（数据库或和风API）。",
        # log_level="ERROR",
        # host="127.0.0.1", port=8002
    )

    # 实例化数据库模式的天气服务（仅在 database 模式下使用）
    service = WeatherService()

    # 打印数据源配置信息
    logger.info(f"天气数据源配置: {conf.weather_source}")

    @weather_mcp.tool(
        name="query_weather",
        description=(
            "查询天气数据。参数：city(城市名称), start_date(开始日期，格式YYYY-MM-DD), "
            "end_date(结束日期，格式YYYY-MM-DD，可选，不传则只查start_date当天)。"
            "示例：query_weather(city='北京', start_date='2025-07-30') 查询北京2025-07-30的天气"
        )
    )
    def query_weather(city: str, start_date: str, end_date: str = None) -> str:
        """
        MCP 工具：参数化天气查询

        这个工具会被 LangChain Agent 自动调用。
        Agent 从用户的自然语言输入中提取参数，然后传给这个函数。

        内部根据配置项 conf.weather_source 决定数据来源：
        - "database"：从 MySQL 数据库查询
        - "api"：直接从和风天气 API 实时获取

        参数：
            city (str): 城市名称，如 "北京"、"上海"
            start_date (str): 开始日期，格式 YYYY-MM-DD
            end_date (str, optional): 结束日期，格式 YYYY-MM-DD（可选）

        返回值：
            str: JSON 字符串，格式统一为：
                {"status": "success", "data": [...]}  # 查询成功
                {"status": "no_data", "message": "..."}  # 无数据
                {"status": "error", "message": "..."}  # 执行出错
        """
        logger.info(f"执行天气查询: city={city}, start_date={start_date}, end_date={end_date}, 数据源={conf.weather_source}")

        if conf.weather_source == "api":
            # 模式1：直接从和风天气 API 获取
            logger.info("使用和风天气 API 模式")
            return fetch_weather_from_api(city, start_date, end_date)
        else:
            # 模式2：从数据库查询（默认）
            logger.info("使用数据库查询模式")
            return service.query_weather(city, start_date, end_date)

    # 打印服务器信息
    logger.info("=== 天气MCP服务器信息 ===")
    logger.info(f"名称: {weather_mcp.name}")
    logger.info(f"描述: {weather_mcp.instructions}")

    # 启动服务器
    try:
        print("服务器已启动，请访问 http://127.0.0.1:8002/mcp")
        weather_mcp.run(transport="http", host="0.0.0.0", port=8002)
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == '__main__':
    create_weather_mcp_server()
