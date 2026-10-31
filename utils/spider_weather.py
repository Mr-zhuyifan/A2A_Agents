"""
需求：实现天气数据爬虫，定期从第三方API获取天气数据并存储到MySQL数据库
思路步骤：
1. 导入必要的模块和库
2. 配置API密钥、城市代码和数据库连接参数
3. 实现fetch_weather_data函数（从API获取天气数据）
4. 实现get_latest_update_time函数（获取数据库中最新更新时间）
5. 实现should_update_data函数（判断是否需要更新数据）
6. 实现store_weather_data函数（将天气数据存储到数据库）
7. 实现update_weather函数（更新指定城市的天气数据）
8. 实现setup_scheduler函数（设置定时任务）
9. 主函数（创建表、执行一次更新、启动定时任务）
"""
import gzip
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import pymysql
import pytz
import requests
import schedule

# 让脚本以 `python utils/spider_weather.py` 直接运行时也能导入项目包
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_dotenv  # noqa: E402  （必须在调整 sys.path 之后导入）

load_dotenv()

# 配置（从环境变量 / .env 读取，参考 .env.example）
API_KEY = os.getenv("QWEATHER_API_KEY", "").strip()
city_codes = {
    "北京": "101010100",
    "上海": "101020100",
    "杭州": "101210101",
    "南京": "101190101",
    "广州": "101280101",
    "深圳": "101280601",
    "成都": "101270101"
}
BASE_URL = os.getenv("QWEATHER_BASE_URL", "https://devapi.qweather.com/v7/weather/30d")
TZ = pytz.timezone('Asia/Shanghai')  # 使用上海时区

# MySQL 配置（从环境变量 / .env 读取）
db_config = {
    "host": os.getenv("MYSQL_HOST", "localhost"),
    "user": os.getenv("MYSQL_USER", "root"),
    "password": os.getenv("MYSQL_PASSWORD", "root"),
    "database": os.getenv("MYSQL_DATABASE", "travel_rag"),
    "charset": "utf8mb4"
}


def connect_db():
    return pymysql.connect(**db_config)


def fetch_weather_data(city, location):
    """
    从第三方API拿取天气数据。
    :param city:
    :param location:
    :return: 天气数据
    """
    if not API_KEY:
        print("未配置和风天气 Key：请在项目根目录 .env 中设置 QWEATHER_API_KEY（参考 .env.example）")
        return None

    headers = {
        "X-QW-Api-Key": API_KEY,
        "Accept-Encoding": "gzip"
    }
    url = f"{BASE_URL}?location={location}"
    try:
        response = requests.get(url, headers=headers, timeout=10)
        print(response)
        # <Response [200]>
        response.raise_for_status()
        print(222222222, response.raise_for_status())
        # 222222222 None
        print(11111111, response.headers.get('Content-Encoding'))
        # 11111111 gzip
        if response.headers.get('Content-Encoding') == 'gzip':
            print("gzip")
            # gzip
            # 使用.decompress方法时候报错了
            data = gzip.decompress(response.content).decode('utf-8')
            print(1111, data)
        else:
            data = response.text

        # exit()
        return json.loads(data)
    except requests.RequestException as e:
        print(f"请求 {city} 天气数据失败: {e}")
        return None
    except json.JSONDecodeError as e:
        print(f"{city} JSON 解析错误: {e}, 响应内容: {response.text[:500]}...")
        return None
    except gzip.BadGzipFile:
        print(f"{city} 数据未正确解压，尝试直接解析: {response.text[:500]}...")
        return json.loads(response.text) if response.text else None


def get_latest_update_time(cursor, city):
    """
    拿取 这个 城市city 在数据表weather_data中最新更新时间。
    :param cursor:
    :param city:
    :return:
    """
    cursor.execute("SELECT MAX(update_time) FROM weather_data WHERE city = %s", (city,))
    result = cursor.fetchone()
    return result[0] if result[0] else None


def should_update_data(latest_time, force_update=False):
    """
    如果没有更新时间 或者 更新时间大于24小时 或者 force_update=True，就返回True，就去更新数据库中的这个城市的天气数据。
    其他情况返回False，表示不更新数据库中的数据。
    :param latest_time: 某个城市最新的更新时间
    :param force_update: 默认否
    :return: 是 或这 否 true or false
    """
    if force_update:
        return True
    if not latest_time:
        return True
    current_time = datetime.now(TZ)
    latest_time = latest_time.replace(tzinfo=TZ)
    return (current_time - latest_time).total_seconds() / 3600 >= 24


def store_weather_data(conn, cursor, city, data):
    """
    把从第三方拿到的天气数据（json）解析出来，组装成可以插入到mysql天气表中的insert语句。然后执行插入动作。
    :param conn:
    :param cursor:
    :param city:
    :param data:
    :return:
    """

    # {"code":"200",
    # "updateTime":"2026-02-04T10:42+08:00",
    # "fxLink":"https://www.qweather.com/weather/beijing-101010100.html",
    # "daily":[
    #     {"fxDate":"2026-02-04","sunrise":"07:21","sunset":"17:37","moonrise":"20:21",
    #     "moonset":"08:41","moonPhase":"亏凸月","moonPhaseIcon":"805","tempMax":"12",
    #     "tempMin":"-2","iconDay":"100","textDay":"晴","iconNight":"150","textNight":"晴",
    #     "wind360Day":"225","windDirDay":"西南风","windScaleDay":"1-3","windSpeedDay":"3",
    #     "wind360Night":"0","windDirNight":"北风","windScaleNight":"1-3","windSpeedNight":"16",
    #     "humidity":"29","precip":"0.0","pressure":"1016","vis":"25","cloud":"0","uvIndex":"3"}
    #     。。。。
    # "refer":{"sources":["QWeather"],"license":["QWeather Developers License"]}}

    if not data or data.get("code") != "200":
        print(f"{city} 数据无效，跳过存储。")
        return

    daily_data = data.get("daily", [])
    update_time = datetime.fromisoformat(data.get("updateTime").replace("+08:00", "+08:00")).replace(tzinfo=TZ)

    for day in daily_data:
        # {"fxDate":"2026-02-04","sunrise":"07:21","sunset":"17:37","moonrise":"20:21",
        #     #     "moonset":"08:41","moonPhase":"亏凸月","moonPhaseIcon":"805","tempMax":"12",
        #     #     "tempMin":"-2","iconDay":"100","textDay":"晴","iconNight":"150","textNight":"晴",
        #     #     "wind360Day":"225","windDirDay":"西南风","windScaleDay":"1-3","windSpeedDay":"3",
        #     #     "wind360Night":"0","windDirNight":"北风","windScaleNight":"1-3","windSpeedNight":"16",
        #     #     "humidity":"29","precip":"0.0","pressure":"1016","vis":"25","cloud":"0","uvIndex":"3"}
        fx_date = datetime.strptime(day["fxDate"], "%Y-%m-%d").date()
        values = (
            city,
            fx_date,
            day.get("sunrise"), day.get("sunset"),
            day.get("moonrise"), day.get("moonset"),
            day.get("moonPhase"), day.get("moonPhaseIcon"),
            day.get("tempMax"), day.get("tempMin"),
            day.get("iconDay"), day.get("textDay"),
            day.get("iconNight"), day.get("textNight"),
            day.get("wind360Day"), day.get("windDirDay"), day.get("windScaleDay"), day.get("windSpeedDay"),
            day.get("wind360Night"), day.get("windDirNight"), day.get("windScaleNight"), day.get("windSpeedNight"),
            day.get("precip"), day.get("uvIndex"),
            day.get("humidity"), day.get("pressure"),
            day.get("vis"), day.get("cloud"),
            update_time
        )
        insert_query = """
        INSERT INTO weather_data (
            city, fx_date, sunrise, sunset, moonrise, moonset, moon_phase, moon_phase_icon,
            temp_max, temp_min, icon_day, text_day, icon_night, text_night,
            wind360_day, wind_dir_day, wind_scale_day, wind_speed_day,
            wind360_night, wind_dir_night, wind_scale_night, wind_speed_night,
            precip, uv_index, humidity, pressure, vis, cloud, update_time
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON DUPLICATE KEY UPDATE
            sunrise = VALUES(sunrise), sunset = VALUES(sunset), moonrise = VALUES(moonrise),
            moonset = VALUES(moonset), moon_phase = VALUES(moon_phase), moon_phase_icon = VALUES(moon_phase_icon),
            temp_max = VALUES(temp_max), temp_min = VALUES(temp_min), icon_day = VALUES(icon_day),
            text_day = VALUES(text_day), icon_night = VALUES(icon_night), text_night = VALUES(text_night),
            wind360_day = VALUES(wind360_day), wind_dir_day = VALUES(wind_dir_day), wind_scale_day = VALUES(wind_scale_day),
            wind_speed_day = VALUES(wind_speed_day), wind360_night = VALUES(wind360_night),
            wind_dir_night = VALUES(wind_dir_night), wind_scale_night = VALUES(wind_scale_night),
            wind_speed_night = VALUES(wind_speed_night), precip = VALUES(precip), uv_index = VALUES(uv_index),
            humidity = VALUES(humidity), pressure = VALUES(pressure), vis = VALUES(vis),
            cloud = VALUES(cloud), update_time = VALUES(update_time)
        """
        try:
            cursor.execute(insert_query, values)
            print(f"{city} {fx_date} 数据写入/更新成功: {day.get('textDay')}, 影响行数: {cursor.rowcount}")
            conn.commit()
            print(f"{city} 事务提交完成。")
        except pymysql.Error as e:
            print(f"{city} {fx_date} 数据库错误: {e}")
            conn.rollback()
            print(f"{city} 事务回滚。")


def update_weather(force_update=False):
    """

    :param force_update:
    :return:
    """
    conn = connect_db()
    cursor = conn.cursor()

    # city_codes = {
    #     "北京": "101010100",
    #     "上海": "101020100",
    #     "广州": "101280101",
    #     "深圳": "101280601"
    # }
    for city, location in city_codes.items():
        # 拿取 这个 城市city 在数据表weather_data中最新更新时间。
        latest_time = get_latest_update_time(cursor, city)
        # 如果没有更新时间 或者 更新时间大于24小时 或者 force_update=True，就返回True，就去更新数据库中的这个城市的天气数据。
        # 其他情况返回False，表示不更新数据库中的数据。
        if should_update_data(latest_time, force_update):
            print(f"开始更新 {city} 天气数据...")
            # 从第三方API拿取天气数据
            data = fetch_weather_data(city, location)
            if data:
                # 把从第三方拿到的天气数据（json）解析出来，组装成可以插入到mysql天气表中的insert语句。然后执行插入动作。
                store_weather_data(conn, cursor, city, data)
        else:
            print(f"{city} 数据已为最新，无需更新。最新更新时间: {latest_time}")

    cursor.close()
    conn.close()


def setup_scheduler():
    # 北京时间 1:00 对应 PDT 前一天的 16:00（夏令时）
    schedule.every().day.at("16:00").do(update_weather)
    while True:
        schedule.run_pending()
        time.sleep(60)


if __name__ == "__main__":
    # 初始检查和更新
    conn = pymysql.connect(**db_config)
    with conn:
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_data (
            id INT AUTO_INCREMENT PRIMARY KEY,
            city VARCHAR(50) NOT NULL COMMENT '城市名称',
            fx_date DATE NOT NULL COMMENT '预报日期',
            sunrise TIME COMMENT '日出时间',
            sunset TIME COMMENT '日落时间',
            moonrise TIME COMMENT '月升时间',
            moonset TIME COMMENT '月落时间',
            moon_phase VARCHAR(20) COMMENT '月相名称',
            moon_phase_icon VARCHAR(10) COMMENT '月相图标代码',
            temp_max INT COMMENT '最高温度',
            temp_min INT COMMENT '最低温度',
            icon_day VARCHAR(10) COMMENT '白天天气图标代码',
            text_day VARCHAR(20) COMMENT '白天天气描述',
            icon_night VARCHAR(10) COMMENT '夜间天气图标代码',
            text_night VARCHAR(20) COMMENT '夜间天气描述',
            wind360_day INT COMMENT '白天风向360角度',
            wind_dir_day VARCHAR(20) COMMENT '白天风向',
            wind_scale_day VARCHAR(10) COMMENT '白天风力等级',
            wind_speed_day INT COMMENT '白天风速 (km/h)',
            wind360_night INT COMMENT '夜间风向360角度',
            wind_dir_night VARCHAR(20) COMMENT '夜间风向',
            wind_scale_night VARCHAR(10) COMMENT '夜间风力等级',
            wind_speed_night INT COMMENT '夜间风速 (km/h)',
            precip DECIMAL(5,1) COMMENT '降水量 (mm)',
            uv_index INT COMMENT '紫外线指数',
            humidity INT COMMENT '相对湿度 (%)',
            pressure INT COMMENT '大气压强 (hPa)',
            vis INT COMMENT '能见度 (km)',
            cloud INT COMMENT '云量 (%)',
            update_time DATETIME COMMENT '数据更新时间',
            UNIQUE KEY unique_city_date (city, fx_date)
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci COMMENT='天气数据表'
        """)
        conn.commit()

    # 立即执行一次更新
    update_weather()
    # 我们全国有2000个城市

    # 启动定时任务
    setup_scheduler()

# 真实应该怎么更新：
# 1 用户来查询某个城市 比如 北京的天气数据。
# 2 去数据库中查询北京天气数据，select weather_content,update_time from weather_data where city = "北京" and date = "2026-03-04"
# 3 检验这条数据的update_time，如果是1小时之前的，就从第三方api拉取最新数据，返回给用户。同时更新mysql数据。
# 4 如果这条数据是1小时之内更新的，就直接返回给用户。
# 5 做一个缓存，缓存的key是 city+date，设置过期时间ttl，当前时间-更新时间（1小时之内）

# 以上是第一种方式
# 其他方式：
# 从redis中查，有就用，没有就查第三方api，缓存redis，设置ttl=1小时。


# {"code":"200",
# "updateTime":"2026-02-04T10:42+08:00",
# "fxLink":"https://www.qweather.com/weather/beijing-101010100.html",
# "daily":[
#     {"fxDate":"2026-02-04","sunrise":"07:21","sunset":"17:37","moonrise":"20:21",
#     "moonset":"08:41","moonPhase":"亏凸月","moonPhaseIcon":"805","tempMax":"12",
#     "tempMin":"-2","iconDay":"100","textDay":"晴","iconNight":"150","textNight":"晴",
#     "wind360Day":"225","windDirDay":"西南风","windScaleDay":"1-3","windSpeedDay":"3",
#     "wind360Night":"0","windDirNight":"北风","windScaleNight":"1-3","windSpeedNight":"16",
#     "humidity":"29","precip":"0.0","pressure":"1016","vis":"25","cloud":"0","uvIndex":"3"}
#     。。。。
# "refer":{"sources":["QWeather"],"license":["QWeather Developers License"]}}
