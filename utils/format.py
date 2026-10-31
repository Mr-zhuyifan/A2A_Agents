"""
需求：实现数据格式化工具，处理日期、时间和数值类型的序列化
思路步骤：
1. 导入必要的模块和库
2. 实现default_encoder函数（格式化单个对象）
3. 创建DateEncoder类（自定义JSON编码器，处理非标准类型序列化）
4. 重写DateEncoder.default方法（处理序列化时的默认对象转换）
"""
import json
from datetime import date, datetime, timedelta
from decimal import Decimal


def default_encoder(obj):  # 定义编码器方法，用于格式化单个对象
    if isinstance(obj, datetime):  # 检查是否为datetime，返回带时间的格式化字符串
        return obj.strftime('%Y-%m-%d %H:%M:%S')
    if isinstance(obj, date):  # 检查是否为date，返回日期格式化字符串
        return obj.strftime('%Y-%m-%d')
    if isinstance(obj, timedelta):  # 检查是否为timedelta，转换为字符串
        return str(obj)
    if isinstance(obj, Decimal):  # 检查是否为Decimal，转换为浮点数
        # TODO 更优的方案 str(obj)
        return float(obj)
    return obj  # 否则返回原对象

# 定义自定义JSON编码器类，继承自json.JSONEncoder，用于处理非标准类型序列化
class DateEncoder(json.JSONEncoder):
    def default(self, obj):  # 重写default方法，处理序列化时的默认对象转换
        if isinstance(obj, (date, datetime)):  # 检查对象是否为date或datetime类型，对于datetime返回带时间的字符串，对于date返回日期字符串
            return obj.strftime('%Y-%m-%d %H:%M:%S') if isinstance(obj, datetime) else obj.strftime('%Y-%m-%d')
        if isinstance(obj, timedelta):  # 检查对象是否为timedelta类型，将时间差转换为字符串
            return str(obj)
        if isinstance(obj, Decimal):  # 检查对象是否为Decimal类型，将Decimal转换为浮点数以兼容JSON
            return float(obj)
        return super().default(obj)  # 对于其他类型，调用父类默认方法