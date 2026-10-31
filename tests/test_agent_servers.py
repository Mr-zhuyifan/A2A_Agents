"""
Agent 服务器测试模块

测试内容：
1. AgentCard 定义验证（纯属性检查，零依赖）
2. Agent 服务器集成测试（需要 MCP 服务器运行 + LLM API 可访问）

运行方式：
    cd SmartVoyage
    python -m tests.test_agent_servers
"""

import unittest
import sys
import os

# 确保能导入 SmartVoyage 模块
# sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


# ==================== 1. AgentCard 定义测试（纯属性检查，零依赖） ====================

class TestWeatherAgentCard(unittest.TestCase):
    """测试天气 Agent 的代理卡片定义"""

    def test_agent_card_basic(self):
        """验证代理卡片基本信息"""
        from a2a_server.weather_server import agent_card

        self.assertEqual(agent_card.name, "WeatherQueryAssistant")
        self.assertIn("天气查询", agent_card.description)
        self.assertIn(agent_card.url, ["http://localhost:5005", "http://127.0.0.1:5005"])
        self.assertEqual(len(agent_card.skills), 1)

    def test_weather_skill_examples(self):
        """验证天气技能示例"""
        from a2a_server.weather_server import agent_card

        skill = agent_card.skills[0]
        self.assertIn("weather", skill.name.lower())
        self.assertIsNotNone(skill.description)
        self.assertGreater(len(skill.examples), 0)


class TestTicketAgentCard(unittest.TestCase):
    """测试票务 Agent 的代理卡片定义"""

    def test_agent_card_skills(self):
        """验证票务代理拥有 6 个技能"""
        from a2a_server.ticket_server import agent_card

        self.assertEqual(agent_card.name, "TicketAssistant")
        self.assertIn(agent_card.url, ["http://localhost:5006", "http://127.0.0.1:5006"])
        self.assertEqual(len(agent_card.skills), 6)

    def test_has_query_and_order_skills(self):
        """验证同时包含查询和预定技能"""
        from a2a_server.ticket_server import agent_card

        skill_names = [s.name for s in agent_card.skills]
        query_skills = [n for n in skill_names if 'query' in n.lower()]
        order_skills = [n for n in skill_names if 'order' in n.lower()]
        self.assertEqual(len(query_skills), 3, "应该有3个查询技能")
        self.assertEqual(len(order_skills), 3, "应该有3个预定技能")


class TestTripAgentCard(unittest.TestCase):
    """测试行程 Agent 的代理卡片定义"""

    def test_agent_card_skills(self):
        """验证行程代理拥有 6 个技能"""
        from a2a_server.trip_server import agent_card

        self.assertEqual(agent_card.name, "TripAssistant")
        self.assertIn(agent_card.url, ["http://localhost:5007", "http://127.0.0.1:5007"])
        self.assertEqual(len(agent_card.skills), 6)

    def test_has_all_trip_skills(self):
        """验证包含租车、旅游团、保险的查询和预定技能"""
        from a2a_server.trip_server import agent_card

        skill_names = [s.name.lower() for s in agent_card.skills]
        self.assertTrue(any('car' in n for n in skill_names), "缺少租车技能")
        self.assertTrue(any('tour' in n for n in skill_names), "缺少旅游团技能")
        self.assertTrue(any('insurance' in n for n in skill_names), "缺少保险技能")


class TestCopywritingAgentCard(unittest.TestCase):
    """测试小红书文案 Agent 的代理卡片定义"""

    def test_agent_card_skills(self):
        """验证文案代理的卡片信息和技能"""
        from a2a_server.copywriting_server import agent_card

        self.assertEqual(agent_card.name, "CopywritingAssistant")
        self.assertIn(agent_card.url, ["http://localhost:5008", "http://127.0.0.1:5008"])
        self.assertEqual(len(agent_card.skills), 3)

    def test_has_copywriting_and_fact_collection_skills(self):
        """验证包含文案撰写和素材收集技能"""
        from a2a_server.copywriting_server import agent_card

        skill_names = [s.name.lower() for s in agent_card.skills]
        self.assertTrue(any('xiaohongshu' in n for n in skill_names), "缺少小红书文案技能")
        self.assertTrue(any('marketing' in n for n in skill_names), "缺少营销文案技能")
        self.assertTrue(any('facts' in n for n in skill_names), "缺少素材收集技能")


# ==================== 2. Agent 服务器集成测试 ====================
# 前置条件：
# 1. 对应的 MCP 服务器已启动（天气 8002、票务 8001、行程 8003）
# 2. LLM API 可访问（配置了有效的 API_KEY）
# 3. 文案代理还要求天气、票务、行程三个 A2A 代理已启动

class TestWeatherAgentIntegration(unittest.TestCase):
    """集成测试：天气 Agent 服务器"""

    def test_handle_task_weather_query(self):
        """测试天气 Agent 完整查询流程"""
        import asyncio
        from a2a_server.weather_server import WeatherQueryServer, query_weather
        from python_a2a import Task, TaskState

        # 直接调用 query_weather 异步函数，验证 MCP + LLM 完整链路
        result = asyncio.run(query_weather("北京 2026-11-10 天气"))
        print(result)
        self.assertEqual(result["status"], "success")
        # 验证返回内容包含城市名
        self.assertIn("北京", result["message"])

    def test_weather_server_instance(self):
        """验证天气服务器实例化"""
        from a2a_server.weather_server import WeatherQueryServer
        server = WeatherQueryServer()
        self.assertEqual(server.agent_card.name, "WeatherQueryAssistant")


class TestTicketAgentIntegration(unittest.TestCase):
    """集成测试：票务 Agent 服务器"""

    def test_handle_task_ticket_query(self):
        """测试票务 Agent 完整查询流程"""
        import asyncio
        from a2a_server.ticket_server import query_tickets

        result = asyncio.run(query_tickets("北京 到 成都 火车票 2026-05-01"))
        self.assertEqual(result["status"], "success")
        # 验证返回内容包含出发和到达城市
        self.assertIn("北京", result["message"])
        self.assertIn("成都", result["message"])

    def test_ticket_server_instance(self):
        """验证票务服务器实例化"""
        from a2a_server.ticket_server import TicketQueryServer
        server = TicketQueryServer()
        self.assertEqual(server.agent_card.name, "TicketAssistant")


class TestTripAgentIntegration(unittest.TestCase):
    """集成测试：行程 Agent 服务器"""

    def test_handle_task_trip_query(self):
        """测试行程 Agent 完整查询流程"""
        import asyncio
        from a2a_server.trip_server import query_trip

        result = asyncio.run(query_trip("成都租车 2026-05-01"))
        self.assertEqual(result["status"], "success")

    def test_trip_server_instance(self):
        """验证行程服务器实例化"""
        from a2a_server.trip_server import TripQueryServer
        server = TripQueryServer()
        self.assertEqual(server.agent_card.name, "TripAssistant")


class TestCopywritingAgentIntegration(unittest.TestCase):
    """集成测试：小红书文案 Agent 服务器"""

    def test_copywriting_server_instance(self):
        """验证文案服务器实例化"""
        from a2a_server.copywriting_server import CopywritingServer
        server = CopywritingServer()
        self.assertEqual(server.agent_card.name, "CopywritingAssistant")


if __name__ == "__main__":
    unittest.main(verbosity=2)
