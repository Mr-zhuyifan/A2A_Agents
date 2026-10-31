"""
需求：实现基于A2A的行程管家服务器，处理用户的租车、旅游团、保险查询及预订请求

架构说明：
    本服务器是 SmartVoyage 系统中的又一个子代理（Sub-Agent），负责处理所有行程相关任务。
    它运行在独立的进程中（127.0.0.1:5007），通过 A2A（Agent2Agent）协议与主助手通信。

    与 ticket_server.py 和 weather_server.py 保持一致的架构模式：
    - 使用 LangChain Agent + MCP Tools 模式
    - LLM 从自然语言中提取参数（城市、日期、类型等）
    - 调用 MCP Server 的参数化工具（端口 8003）
    - MCP Server 返回结果后，LLM 将结果格式化为友好的中文回复

    工作流程：
    1. 主助手通过 A2A 协议向本服务器发送任务（Task）
    2. 本服务器收到任务后，提取用户的自然语言查询
    3. 使用 LangChain Agent 处理查询：
       a. LLM 分析用户输入，决定调用哪个 MCP 工具
       b. LangChain 自动调用 MCP Server 的工具（端口 8003）
       c. 工具返回结果后，LLM 将结果格式化为友好的中文回复
    4. 将结果返回给主助手

    MCP Server（端口 8003）提供的工具包括：
    - 租车查询工具、旅游团查询工具、保险查询工具
    - 租车预订工具、旅游团报名工具、保险购买工具
"""

# ==================== 导入依赖 ====================
import os  # 环境变量
import json  # JSON 处理
import asyncio  # 异步 IO 库

from mcp import ClientSession  # MCP 客户端会话
from mcp.client.streamable_http import streamablehttp_client  # MCP HTTP 流式客户端
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState
# A2AServer: A2A 服务器基类
# run_server: 启动服务器的函数
# AgentCard: 代理卡片
# AgentSkill: 代理技能
# TaskStatus: 任务状态对象
# TaskState: 任务状态枚举（COMPLETED / FAILED / INPUT_REQUIRED）

from langchain_openai import ChatOpenAI  # LangChain 的大模型接口
from langchain_core.prompts import ChatPromptTemplate  # 提示模板
from langchain_mcp_adapters.tools import load_mcp_tools  # 从 MCP 会话加载工具
from langchain.agents import create_tool_calling_agent, AgentExecutor
# create_tool_calling_agent: 创建基于工具调用的 Agent
# AgentExecutor: Agent 执行器，负责运行 Agent 循环
from fastmcp import Client
from datetime import datetime  # 时间处理
import pytz  # 时区库

from config import Config  # 项目配置
from create_logger import logger  # 日志模块

conf = Config()  # 全局配置实例

# ==================== 初始化大模型 ====================
# 创建一个 LLM 实例，供 Agent 使用
llm = ChatOpenAI(
    **conf.llm_config
)

# ==================== MCP 客户端配置 ====================
# MCP Server 运行在 8003 端口，提供行程相关的工具（查询 + 预订）
# 支持本地和 Docker 环境自动适配
MCP_URL = os.getenv("TRIP_MCP_URL", "http://127.0.0.1:8003/mcp")
# 每次查询使用独立 Client，避免 Flask 多线程 / 多事件循环共享会话导致卡死
MCP_TIMEOUT = int(os.getenv("TRIP_MCP_TIMEOUT", "45"))

# ==================== 行程查询函数 ====================
async def query_trip(conversation: str) -> dict:
    """
    通过 LangChain Agent + MCP Tools 执行行程查询或预订

    与 ticket_server.py 的 query_tickets 使用完全相同的架构模式：
    1. 连接 MCP Server 加载所有可用的工具
    2. 创建一个 LangChain Agent，让它自主选择调用哪个工具
    3. AgentExecutor 负责执行 Agent 循环

    参数：
        conversation (str): 用户的查询内容，例如：
            "我想租一辆车，北京，明天"
            "有没有去丽江的旅游团"
            "买一份旅行保险"
            "帮我报个北京三日游的团"

    返回值：
        dict: 查询结果，格式为：
            - {"status": "success", "message": "格式化后的查询结果"}
            - {"status": "error", "message": "错误信息"}
    """
    try:
        async with Client(MCP_URL, timeout=MCP_TIMEOUT) as mcp_client:
        # 连接 MCP Server（端口 8003），建立通信通道
        # async with streamablehttp_client(MCP_URL) as (read, write, _):
        #     async with ClientSession(read, write) as session:
        #         await session.initialize()  # 初始化会话

                # 从 MCP 会话中加载所有可用工具
                tools = await load_mcp_tools(mcp_client.session)

                # 定义 Agent 的系统 Prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个行程管家助手，能够调用工具来完成租车查询/预定、旅游团查询/报名、保险查询/购买。
你需要仔细分析用户的问题，从问题中提取工具需要的参数，然后调用对应的工具。
如果用户提供的信息不足以提取到调用工具所有必要参数，则向用户追问，以获取该信息。不能自己编撰参数。

注意：
- query_tour_group 使用语义搜索，query_text 参数是用户的自然语言查询描述（如"想看雪山的地方"），city 参数是可选的城市过滤。
- query_car_rental 需要取车城市、还车城市、日期。
- query_insurance 可以直接调用，不传 insurance_type 会返回所有保险。

查询到结果后，请用清晰的中文格式化输出，格式如下：
- 租车：取车城市 到 还车城市 日期: 车型XX，公司XX，价格XX元/天，余量XX
- 旅游团：城市 团名: 天数XX天，价格XX元，余位XX/总数XX，亮点XX，出发日期列表
- 保险：保险类型: 名称XX，价格XX元/份，保障范围XX，保险公司XX
预定成功后请直接告知用户预定结果。
当前日期是{current_date}。"""),
                    ("human", "{input}"),  # 用户输入会替换到这里
                    ("placeholder", "{agent_scratchpad}"),  # Agent 思考过程的占位符
                ])

                # 创建基于工具调用的 Agent
                agent = create_tool_calling_agent(llm, tools, prompt)

                # 创建 Agent 执行器
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                # 获取当前日期，注入到 Prompt 中
                current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')

                # 执行 Agent
                response = await agent_executor.ainvoke({
                    "input": conversation,
                    "current_date": current_date
                })

                return {"status": "success", "message": response["output"]}

    except ExceptionGroup as eg:
        # MCP 工具内部使用 anyio.create_task_group()，失败时抛出 ExceptionGroup
        first_exc = eg.exceptions[0] if eg.exceptions else eg
        logger.error(f"行程 MCP 查询出错：{first_exc}")
        return {"status": "error", "message": f"行程 MCP 查询出错：{first_exc}"}
    except BaseException as e:
        # 兜底捕获（包括 ExceptionGroup 等 BaseException 子类）
        logger.error(f"行程 MCP 查询出错：{str(e)}")
        return {"status": "error", "message": f"行程 MCP 查询出错：{str(e)}"}


# ==================== Agent Card（代理卡片） ====================
# 描述行程管家代理的能力、技能等信息
agent_card = AgentCard(
    name="TripAssistant",  # 代理名称
    description="基于 LangChain 提供行程管家服务的统一助手，支持租车、旅游团、保险的查询与预订",  # 代理描述
    url="http://127.0.0.1:5007",  # 代理的访问地址
    version="1.0.0",  # 版本号
    capabilities={"streaming": True, "memory": True},  # 支持的能力
    skills=[
        # 技能1：租车查询
        AgentSkill(
            name="query car rental",
            description="查询租车信息，支持指定取车城市、还车城市、日期和车型",
            examples=["租车 北京 上海 2025-08-01", "北京租一辆SUV 明天"]
        ),
        # 技能2：旅游团查询（语义搜索）
        AgentSkill(
            name="query tour group",
            description="通过语义搜索查询旅游团信息，支持自然语言描述需求（如想看雪山的地方）和可选的城市过滤",
            examples=["想看雪山的地方", "适合亲子游的短途旅行", "美食之旅", "北京的旅游团"]
        ),
        # 技能3：保险查询
        AgentSkill(
            name="query insurance",
            description="查询旅行保险产品，支持指定保险类型和日期",
            examples=["旅行保险 2025-08-01", "买一份综合型旅行保险"]
        ),
        # 技能4：租车预订
        AgentSkill(
            name="order car rental",
            description="根据日期、车型和数量预订租车服务",
            examples=["预订租车 2025-08-01 SUV 1辆"]
        ),
        # 技能5：旅游团报名
        AgentSkill(
            name="order tour group",
            description="根据日期、团名和人数报名旅游团",
            examples=["报名丽江三日游 2025-08-01 2人"]
        ),
        # 技能6：保险购买
        AgentSkill(
            name="order insurance",
            description="根据保险类型、日期和份数购买旅行保险",
            examples=["购买综合型保险 2025-08-01 1份"]
        )
    ]
)


# ==================== 行程查询服务器类 ====================
class TripQueryServer(A2AServer):
    """
    行程管家 A2A 服务器

    这个类继承自 A2AServer，负责处理所有行程相关的任务。
    工作流程与 TicketQueryServer 完全一致：
    1. 接收来自主助手的任务（Task）
    2. 从任务中提取用户的查询内容
    3. 调用 query_trip 函数执行查询
    4. 将查询结果设置到任务状态中，返回给主助手
    """

    def __init__(self):
        """
        初始化行程查询服务器
        """
        super().__init__(agent_card=agent_card)  # 调用父类初始化，注册 Agent Card

    def handle_task(self, task):
        """
        处理来自 A2A 客户端的任务 —— 本服务器的核心方法

        完整流程：
        1. 提取输入：从任务消息中获取用户的查询内容
        2. 调用 MCP 查询：通过 query_trip 函数执行查询
        3. 根据查询结果设置任务状态：
           - 成功：COMPLETED 状态，结果放在 task.artifacts
           - 需要输入：INPUT_REQUIRED 状态，追问信息放在 task.status.message
           - 失败：FAILED 状态，错误信息放在 task.status.message

        参数：
            task: A2A 任务对象

        返回值：
            task: 处理后的任务对象（已设置状态和结果）
        """
        # ========== 步骤1：提取输入 ==========
        content = (task.message or {}).get("content", {})
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        logger.info(f"用户查询: {conversation}")

        try:
            # ========== 步骤2：调用 MCP 查询 ==========
            trip_result = asyncio.run(query_trip(conversation))
            logger.info(f"MCP 查询返回: {trip_result}")

            # ========== 步骤3：根据结果设置任务状态 ==========
            if trip_result.get("status") == "success":
                result_text = trip_result.get("message", "")

                # 检查是否是追问消息
                if "请提供" in result_text or "请确认" in result_text:
                    task.status = TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message={"role": "agent", "content": {"text": result_text}}
                    )
                else:
                    task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)

            elif trip_result.get("status") == "error":
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": trip_result.get("message", "查询失败，请重试。")}}
                )
            else:
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": "查询失败，请重试或提供更多细节。"}}
                )

            return task

        except Exception as e:
            logger.error(f"查询失败: {str(e)}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={"role": "agent",
                         "content": {"text": f"查询失败: {str(e)} 请重试或提供更多细节。"}}
            )
            return task


# ==================== 主函数：启动服务器 ====================
if __name__ == "__main__":
    # 创建行程查询服务器实例
    trip_server = TripQueryServer()

    # 打印服务器信息
    print("\n=== 服务器信息 ===")
    print(f"名称: {trip_server.agent_card.name}")
    print(f"描述: {trip_server.agent_card.description}")
    print("\n技能:")
    for skill in trip_server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")

    # 启动 A2A 服务器，监听 5007 端口
    run_server(trip_server, host="0.0.0.0", port=5007)
