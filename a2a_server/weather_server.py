"""
需求：实现基于A2A的天气查询服务器，处理用户的天气查询请求并返回结果

架构说明：
    本服务器是 SmartVoyage 系统中的一个子代理（Sub-Agent），负责处理天气查询任务。
    它运行在独立的进程中（127.0.0.1:5005），通过 A2A（Agent2Agent）协议与主助手通信。

    工作流程（与票务 Agent 保持一致的架构模式）：
    1. 主助手通过 A2A 协议向本服务器发送任务（Task）
    2. 本服务器收到任务后，提取用户的自然语言查询
    3. 使用 LangChain Agent（基于工具调用的 Agent）处理查询：
       a. LLM 分析用户输入，从自然语言中提取参数（城市、日期等）
       b. 调用 MCP Server 的参数化工具（query_weather）
       c. MCP Server 内部根据参数拼接 SQL 并执行查询
       d. 工具返回结果后，LLM 将结果格式化为友好的中文回复
    4. 将结果返回给主助手

    与旧模式（Text-to-SQL）的区别：
    - 旧模式：LLM 直接生成 SQL 语句 → MCP 执行原始 SQL
      问题：SQL 注入风险、LLM 生成的 SQL 可能带有代码块标记、容易出错
    - 新模式：LLM 从自然语言中提取参数（city、date）→ MCP 内部拼 SQL
      优势：参数化查询（安全）、SQL 由 MCP 统一管理（可控）、容错性好

    涉及的关键技术：
    - LangChain Agent: 让 LLM 自主选择和使用工具的框架
    - Tool Calling Agent: LLM 以结构化格式调用工具
    - MCP Tools: MCP Server 提供的参数化工具（city、start_date、end_date）
    - AgentExecutor: 负责运行 Agent 循环（思考→调用工具→处理结果→继续）
"""

# ==================== 导入依赖 ====================
import os  # 环境变量
import json  # JSON 处理
import asyncio  # 异步 IO 库，用于在同步方法中调用异步 MCP 客户端
from fastmcp import Client
from mcp import ClientSession  # MCP 客户端会话
from mcp.client.streamable_http import streamablehttp_client  # MCP HTTP 流式客户端
from python_a2a import A2AServer, run_server, AgentCard, AgentSkill, TaskStatus, TaskState
# A2AServer: A2A 服务器基类，我们需要继承它来实现自己的天气查询服务器
# run_server: 启动 A2A 服务器的函数
# AgentCard: 代理卡片，描述本代理的能力、技能等信息
# AgentSkill: 代理技能，描述本代理能做什么
# TaskStatus: 任务状态，标记任务是完成、失败还是需要输入
# TaskState: 任务状态枚举（COMPLETED / FAILED / INPUT_REQUIRED）

from langchain_openai import ChatOpenAI  # LangChain 的大模型接口
from langchain_core.prompts import ChatPromptTemplate  # LangChain 的提示模板
from langchain_mcp_adapters.tools import load_mcp_tools  # 从 MCP 会话加载工具
from langchain.agents import create_tool_calling_agent, AgentExecutor
# create_tool_calling_agent: 创建基于工具调用的 Agent
# AgentExecutor: Agent 执行器，负责运行 Agent 循环

from config import Config  # 项目配置（模型地址、API Key等）
from datetime import datetime  # 时间处理，用于获取当前日期
import pytz  # 时区库，用于转换到 Asia/Shanghai 时区

from create_logger import logger  # 日志模块

conf = Config()  # 全局配置实例

# ==================== 初始化大模型 ====================
# 创建一个 LLM 实例，供 Agent 使用
llm = ChatOpenAI(
    **conf.llm_config
)

# ==================== MCP 客户端配置 ====================
# MCP Server 运行在 8002 端口，提供天气查询的参数化工具
# 本服务器通过连接 MCP Server，让 Agent 自主调用其工具
# 支持本地和 Docker 环境自动适配
MCP_URL = os.getenv("WEATHER_MCP_URL", "http://127.0.0.1:8002/mcp")
# 每次查询使用独立 Client，避免 Flask 多线程 / 多事件循环共享会话导致卡死
MCP_TIMEOUT = int(os.getenv("WEATHER_MCP_TIMEOUT", "45"))


# ==================== 天气查询函数 ====================
async def query_weather(conversation: str) -> dict:
    """
    通过 LangChain Agent + MCP Tools 执行天气查询

    与票务 Agent（ticket_server.py）使用完全相同的架构模式：
    1. 连接 MCP Server 加载所有可用的工具
    2. 创建一个 LangChain Agent，让它自主选择调用哪个工具
    3. AgentExecutor 负责执行 Agent 循环：
       - LLM 分析用户输入，从自然语言中提取参数（城市、日期等）
       - 调用 MCP Server 的参数化工具（如 query_weather(city="北京", start_date="2025-07-30")）
       - MCP Server 内部根据参数拼接 SQL 并执行
       - 工具返回结果后，LLM 将结果格式化为友好的中文回复

    与旧模式（Text-to-SQL）的对比：
    旧模式流程：
        用户输入 "北京明天天气" → LLM 生成 SQL → MCP 执行原始 SQL
    新模式流程：
        用户输入 "北京明天天气" → LLM 提取 city="北京", date="明天" →
        MCP 工具接收参数 → MCP 内部拼 SQL → 执行 → LLM 格式化回复

    参数：
        conversation (str): 用户的查询内容，例如：
            "北京明天天气怎么样？"
            "上海未来3天的天气"
            "北京2025-07-30的天气"

    返回值：
        dict: 查询结果，格式为：
            - {"status": "success", "message": "格式化后的查询结果"}  # 查询成功
            - {"status": "error", "message": "错误信息"}  # 查询失败
    """
    try:
        async with Client(MCP_URL, timeout=MCP_TIMEOUT) as mcp_client:
            # 连接 MCP Server（端口 8002），建立通信通道
            # async with streamablehttp_client(MCP_URL) as (read, write, _):
            #     async with ClientSession(read, write) as session:
            #         await session.initialize()  # 初始化会话

            # 从 MCP 会话中加载所有可用工具
            # load_mcp_tools 会自动将 MCP Server 提供的工具转换为 LangChain 工具对象
            # 当前只有一个工具：query_weather(city, start_date, end_date)
            tools = await load_mcp_tools(mcp_client.session)
            print("当前工具")
            for tool in tools:
                print(tool.name)

            # 定义 Agent 的系统 Prompt
            # 这个 Prompt 告诉 LLM：你是谁、你能做什么、如何提取参数、输出格式要求
            prompt = ChatPromptTemplate.from_messages([
                ("system", """你是一个天气查询助手，能够调用工具来查询天气信息。
你需要仔细分析用户的问题，从问题中提取工具需要的参数（城市、日期等），然后调用对应的查询工具。
如果用户提供的信息不足以提取到调用工具的所有必要参数，则向用户追问，以获取该信息。不能自己编撰参数。
注意：
- 用户可能使用相对时间，如"明天"、"后天"、"今天"、"未来3天"等，请根据当前日期转换为具体日期（YYYY-MM-DD格式）。
- 如果用户只给了城市没给日期，需要追问日期。
- 如果用户只给了日期没给城市，需要追问城市。
查询到结果后，请用清晰的中文格式化输出天气信息，包括城市、日期、天气状况、温度、湿度、风向、降水量等。
如果未查到数据，请回复"未找到相关天气数据，请确认或修改查询条件。"
当前日期是{current_date}。"""),
                ("human", "{input}"),  # 用户输入会替换到这里
                ("placeholder", "{agent_scratchpad}"),  # Agent 思考过程的占位符
            ])

            # 创建基于工具调用的 Agent
            # create_tool_calling_agent 会创建一个能自主选择和使用工具的 Agent
            # LLM 以结构化格式（函数调用）来使用工具，而不是用自然语言描述
            agent = create_tool_calling_agent(llm, tools=tools, prompt=prompt)

            # 创建 Agent 执行器
            # AgentExecutor 负责运行 Agent 循环：
            # 1. 将用户输入发给 LLM
            # 2. LLM 决定调用哪个工具 + 提供参数
            # 3. 执行工具调用
            # 4. 将工具结果返回给 LLM
            # 5. LLM 生成最终回复
            agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

            # 获取当前日期，注入到 Prompt 中（用户可能说"明天"、"后天"等相对时间）
            current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')

            # 执行 Agent，传入用户输入和当前日期
            response = await agent_executor.ainvoke({
                "input": conversation,  # 用户输入的查询内容
                "current_date": current_date  # 当前日期，用于相对时间转换
            })

            # AgentExecutor 的 response["output"] 就是 LLM 生成的最终回复
            return {"status": "success", "message": response["output"]}

    except Exception as e:
        # MCP 连接失败、工具调用异常等情况的捕获
        logger.error(f"天气 MCP 查询出错：{str(e)}")
        return {"status": "error", "message": f"天气 MCP 查询出错：{str(e)}"}


# ==================== Agent Card（代理卡片） ====================
# Agent Card 是 A2A 协议中描述一个代理能力的元数据
# 它告诉主助手：我是谁、我能做什么、我运行在哪里、我有哪些技能
# 主助手通过这个卡片来决定是否将任务路由到这个代理
agent_card = AgentCard(
    name="WeatherQueryAssistant",  # 代理名称
    description="基于LangChain提供天气查询服务的助手",  # 代理描述
    url="http://127.0.0.1:5005",  # 代理的访问地址
    version="2.0.0",  # 版本号（升级为2.0.0，表示架构变更）
    capabilities={"streaming": True, "memory": True},  # 支持的能力：流式输出、记忆
    skills=[  # 技能列表：描述本代理具体能做什么
        AgentSkill(
            name="execute weather query",  # 技能名称
            description="执行天气查询，返回天气数据库结果，支持自然语言输入，支持单天和范围查询",  # 技能描述
            examples=["北京 2025-07-30 天气", "上海未来5天", "今天天气如何"]  # 使用示例
        )
    ]
)


# ==================== 天气查询服务器类 ====================
class WeatherQueryServer(A2AServer):
    """
    天气查询 A2A 服务器

    这个类继承自 A2AServer，实现了天气查询的完整流程。
    与旧模式（Text-to-SQL）不同，现在使用 LangChain Agent + MCP Tools 模式：
    - LLM 负责从自然语言中提取参数（城市、日期）
    - MCP Server 内部根据参数拼接 SQL 并执行（参数化查询，更安全）

    任务状态说明：
    - COMPLETED: 任务成功完成，结果放在 task.artifacts 中
    - FAILED: 任务失败，错误信息放在 task.status.message 中
    - INPUT_REQUIRED: 需要用户补充输入，追问信息放在 task.status.message 中
    """

    def __init__(self):
        """
        初始化天气查询服务器

        调用父类的 __init__ 注册 Agent Card。
        """
        super().__init__(agent_card=agent_card)  # 调用父类初始化，注册 Agent Card

    def handle_task(self, task):
        """
        处理来自 A2A 客户端的任务 —— 本服务器的核心方法

        当主助手通过 A2A 协议发送任务过来时，这个方法会被调用。
        它负责完成从"接收任务"到"返回结果"的完整流程。

        完整流程：
        1. 提取输入：从任务消息中获取用户的查询内容
        2. 调用 MCP 查询：通过 query_weather 函数执行查询
           - query_weather 内部使用 LangChain Agent + MCP Tools
           - LLM 自动从自然语言中提取参数（城市、日期）
           - MCP Server 根据参数拼接 SQL 并执行
           - LLM 将结果格式化为友好的中文回复
        3. 根据查询结果设置任务状态：
           - 成功：COMPLETED 状态，结果放在 task.artifacts
           - 需要输入：INPUT_REQUIRED 状态，追问信息放在 task.status.message
           - 失败：FAILED 状态，错误信息放在 task.status.message

        参数：
            task: A2A 任务对象，包含：
                - task.message: 客户端发送的消息（包含用户输入）
                - task.artifacts: 用于存放任务结果（输出）
                - task.status: 任务状态（完成/失败/需要输入）

        返回值：
            task: 处理后的任务对象（已设置状态和结果）
        """
        # ========== 步骤1：提取输入 ==========
        # 从任务消息中获取内容（A2A 协议中，消息以字典格式存储）
        content = (task.message or {}).get("content", {})
        # 提取对话内容，即客户端发起的任务中的用户输入
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        logger.info(f"对话历史及用户问题: {conversation}")

        try:
            # ========== 步骤2：调用 MCP 查询 ==========
            # query_weather 内部会：
            # 1. 连接 MCP Server 加载所有工具
            # 2. 创建 LangChain Agent
            # 3. Agent 自动从自然语言中提取参数（城市、日期等）
            # 4. 调用 MCP 的参数化工具（query_weather(city="北京", start_date="2025-07-30")）
            # 5. MCP Server 内部拼接 SQL 并执行
            # 6. LLM 将结果格式化为友好的中文回复
            weather_result = asyncio.run(query_weather(conversation))
            logger.info(f"MCP 查询返回: {weather_result}")

            # ========== 步骤3：根据结果设置任务状态 ==========
            if weather_result.get("status") == "success":
                result_text = weather_result.get("message", "")

                # 检查是否是追问消息（LLM 发现信息不足时会追问）
                if "请提供" in result_text or "请确认" in result_text:
                    # 需要用户补充信息
                    task.status = TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message={"role": "agent", "content": {"text": result_text}}
                    )
                else:
                    # 查询成功，将结果放入任务产物
                    task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)

            elif weather_result.get("status") == "error":
                # MCP 查询出错：设置为失败状态
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": weather_result.get("message", "查询失败，请重试。")}}
                )
            else:
                # 未知的状态码，也视为失败
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": "查询失败，请重试或提供更多细节。"}}
                )

            return task

        except Exception as e:
            # 捕获所有异常，确保即使出错也能返回有效的任务状态
            logger.error(f"查询失败: {str(e)}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={"role": "agent",
                         "content": {"text": f"查询失败: {str(e)} 请重试或提供更多细节。"}}
            )
            return task


# ==================== 主函数：启动服务器 ====================
if __name__ == "__main__":
    # 创建天气查询服务器实例
    weather_server = WeatherQueryServer()

    # 打印服务器信息，方便确认启动状态
    print("\n=== 服务器信息 ===")
    print(f"名称: {weather_server.agent_card.name}")
    print(f"描述: {weather_server.agent_card.description}")
    print("\n技能:")
    for skill in weather_server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")

    # 启动 A2A 服务器，监听 5005 端口
    # 主助手会通过 http://127.0.0.1:5005 连接本服务器
    run_server(weather_server, host="0.0.0.0", port=5005)
