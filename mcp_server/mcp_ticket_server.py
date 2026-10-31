"""
需求：实现基于MCP的票务查询与预定服务器，提供火车票、机票和演唱会票的查询及预定功能
思路步骤：
1. 导入必要的模块和库
2. 初始化配置和日志记录器
3. 创建TicketService类（封装数据库操作逻辑）
4. 实现参数化查询方法（query_train、query_flight、query_concert）
5. 定义三个票务预定工具函数
6. 创建create_ticket_mcp_server函数（创建MCP服务器，注册所有工具）
7. 主函数（启动MCP服务器）
"""
import pymysql
import json
from datetime import date, datetime, timedelta
from decimal import Decimal
from fastmcp import FastMCP
from config import Config
from create_logger import logger
from utils.format import DateEncoder, default_encoder

conf = Config()


# 票务服务类
class TicketService:
    # FastMCP 会在线程池中并发执行同步工具，MySQL 连接不能跨线程共享；
    # 因此每次查询独立建立连接，避免并发工具调用把同一个连接卡死。
    def _connect(self):
        return pymysql.connect(
            host=conf.host,
            user=conf.user,
            password=conf.password,
            database=conf.database,
            autocommit=True
        )

    # 执行SQL查询，返回JSON字符串
    def _execute_query(self, sql: str, params: list = None) -> str:
        conn = None
        cursor = None
        try:
            conn = self._connect()
            cursor = conn.cursor(pymysql.cursors.DictCursor)
            cursor.execute(sql, params or [])
            results = cursor.fetchall()
            # 格式化结果
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (date, datetime, timedelta, Decimal)):
                        result[key] = default_encoder(value)
            return json.dumps(
                {"status": "success", "data": results} if results
                else {"status": "no_data", "message": "未找到票务数据，请确认查询条件。"},
                cls=DateEncoder, ensure_ascii=False
            )
        except Exception as e:
            logger.error(f"票务查询错误: {str(e)}")
            return json.dumps({"status": "error", "message": str(e)}, ensure_ascii=False)
        finally:
            if cursor is not None:
                cursor.close()
            if conn is not None:
                conn.close()

    # 查询火车票
    def query_train(self, departure_city: str, arrival_city: str, date: str, seat_type: str = None) -> str:
        sql = ("SELECT id, departure_city, arrival_city, departure_time, arrival_time, "
               "train_number, seat_type, price, remaining_seats FROM train_tickets "
               "WHERE departure_city = %s AND arrival_city = %s AND DATE(departure_time) = %s")
        params = [departure_city, arrival_city, date]
        if seat_type:
            sql += " AND seat_type = %s"
            params.append(seat_type)
        return self._execute_query(sql, params)

    # 查询机票
    def query_flight(self, departure_city: str, arrival_city: str, date: str, cabin_type: str = None) -> str:
        sql = ("SELECT id, departure_city, arrival_city, departure_time, arrival_time, "
               "flight_number, cabin_type, price, remaining_seats FROM flight_tickets "
               "WHERE departure_city = %s AND arrival_city = %s AND DATE(departure_time) = %s")
        params = [departure_city, arrival_city, date]
        if cabin_type:
            sql += " AND cabin_type = %s"
            params.append(cabin_type)
        return self._execute_query(sql, params)

    # 查询演唱会票
    def query_concert(self, city: str, artist: str = None, date: str = None,
                      ticket_type: str = None) -> str:
        sql = ("SELECT id, artist, city, venue, start_time, end_time, "
               "ticket_type, price, remaining_seats FROM concert_tickets "
               "WHERE city = %s")
        params = [city]
        if artist:
            sql += " AND artist = %s"
            params.append(artist)
        if date:
            sql += " AND DATE(start_time) = %s"
            params.append(date)
        if ticket_type:
            sql += " AND ticket_type = %s"
            params.append(ticket_type)
        return self._execute_query(sql, params)


# 创建票务MCP服务器
def create_ticket_mcp_server():
    # 创建FastMCP实例
    ticket_mcp = FastMCP(name="TicketTools",
                         instructions="票务查询与预定工具，基于 train_tickets, flight_tickets, concert_tickets 表。支持查询和预定。",
                         # log_level="ERROR",
                         # host="127.0.0.1", port=8001
                         )

    # 实例化票务服务对象
    service = TicketService()

    @ticket_mcp.tool(
        name="query_train",
        description="查询火车票，参数：departure_city(出发城市), arrival_city(到达城市), date(日期，格式YYYY-MM-DD), seat_type(座位类型，可选)"
    )
    def query_train(departure_city: str, arrival_city: str, date: str, seat_type: str = None) -> str:
        logger.info(f"查询火车票: {departure_city} -> {arrival_city}, {date}, {seat_type}")
        return service.query_train(departure_city, arrival_city, date, seat_type)

    @ticket_mcp.tool(
        name="query_flight",
        description="查询机票，参数：departure_city(出发城市), arrival_city(到达城市), date(日期，格式YYYY-MM-DD), cabin_type(舱位类型，可选)"
    )
    def query_flight(departure_city: str, arrival_city: str, date: str, cabin_type: str = None) -> str:
        logger.info(f"查询机票: {departure_city} -> {arrival_city}, {date}, {cabin_type}")
        return service.query_flight(departure_city, arrival_city, date, cabin_type)

    @ticket_mcp.tool(
        name="query_concert",
        description="查询演唱会票，参数：city(城市，必填), artist(艺人，可选), date(日期，格式YYYY-MM-DD，可选), ticket_type(票类型，可选)"
    )
    def query_concert(city: str, artist: str = None, date: str = None, ticket_type: str = None) -> str:
        logger.info(f"查询演唱会票: {city}, {artist}, {date}, {ticket_type}")
        return service.query_concert(city, artist, date, ticket_type)

    @ticket_mcp.tool(
        name="order_train",
        description="根据时间、车次、座位类型、数量预定火车票"
    )
    def order_train(departure_date: str, train_number: str, seat_type: str, number: int) -> str:
        logger.info(f"正在订购火车票: {departure_date}, {train_number}, {seat_type}, {number}")
        logger.info(f"恭喜，火车票预定成功！")
        return "恭喜，火车票预定成功！"

    @ticket_mcp.tool(
        name="order_flight",
        description="根据时间、班次、座位类型、数量预定飞机票"
    )
    def order_flight(departure_date: str, flight_number: str, seat_type: str, number: int) -> str:
        logger.info(f"正在订购飞机票: {departure_date}, {flight_number}, {seat_type}, {number}")
        logger.info(f"恭喜，飞机票预定成功！")
        return "恭喜，飞机票预定成功！"

    @ticket_mcp.tool(
        name="order_concert",
        description="根据时间、明星、场地、座位类型、数量预定演出票"
    )
    def order_concert(start_date: str, aritist: str, venue: str, seat_type: str, number: int) -> str:
        logger.info(f"正在订购演出票: {start_date}, {aritist}, {venue}, {seat_type}, {number}")
        logger.info(f"恭喜，演出票预定成功！")
        return "恭喜，演出票预定成功！"

    # 打印服务器信息
    logger.info("=== 票务MCP服务器信息 ===")
    logger.info(f"名称: {ticket_mcp.name}")
    logger.info(f"描述: {ticket_mcp.instructions}")

    # 运行服务器
    try:
        print("服务器已启动，请访问 http://127.0.0.1:8001/mcp")
        ticket_mcp.run(transport="http", host="0.0.0.0", port=8001)
    except Exception as e:
        print(f"服务器启动失败: {e}")


if __name__ == '__main__':
    create_ticket_mcp_server()
