"""
需求：实现基于A2A的统一票务服务器，处理用户的火车票、机票和演唱会票查询及预订请求

架构说明：
    本服务器是 SmartVoyage 系统中的另一个子代理（Sub-Agent），负责处理所有票务相关任务。
    它运行在独立的进程中（127.0.0.1:5006），通过 A2A（Agent2Agent）协议与主助手通信。

    与 weather_server 的区别：
    - weather_server 使用 LLM 生成 SQL，再通过 MCP 执行（Text-to-SQL 模式）
    - ticket_server 使用 LangChain Agent + MCP Tools 模式（工具调用模式）
      LLM 直接决定调用哪个 MCP 工具（火车票查询/机票查询/演唱会查询/预定），
      然后由 LangChain 的 AgentExecutor 自动完成工具调用和结果处理

    工作流程：
    1. 主助手通过 A2A 协议向本服务器发送任务（Task）
    2. 本服务器收到任务后，提取用户的自然语言查询
    3. 使用 LangChain Agent（基于工具调用的 Agent）处理查询：
       a. LLM 分析用户输入，决定调用哪个 MCP 工具
       b. LangChain 自动调用 MCP Server 的工具（端口 8001）
       c. 工具返回结果后，LLM 将结果格式化为友好的中文回复
    4. 将结果返回给主助手

    涉及的关键技术：
    - LangChain Agent: 一种让 LLM 自主选择和使用工具的框架
    - Tool Calling Agent: 一种 Agent 类型，LLM 以结构化格式调用工具
    - MCP Tools: MCP Server 提供的工具，本系统中包括火车票查询、机票查询等
    - AgentExecutor: LangChain 的执行器，负责运行 Agent 循环（思考→调用工具→处理结果→继续）

    MCP Server（端口 8001）提供的工具包括：
    - 火车票查询工具
    - 机票查询工具
    - 演唱会门票查询工具
    - 火车票预定工具
    - 机票预定工具
    - 演唱会门票预定工具
"""

# ==================== 导入依赖 ====================
import os  # 环境变量
import json  # JSON处理
import asyncio  # 异步IO库

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
# 注意：这里的 temperature 使用配置中的值（默认 0.1）
llm = ChatOpenAI(
    **conf.llm_config
)

# ==================== MCP 客户端配置 ====================
# MCP Server 运行在 8001 端口，提供所有票务相关的工具（查询 + 预定）
# 本服务器通过一个统一的客户端连接 MCP Server，然后让 Agent 自主选择合适的工具
# 支持本地和 Docker 环境自动适配
MCP_URL = os.getenv("TICKET_MCP_URL", "http://127.0.0.1:8001/mcp")
# MCP 调用超时（秒）：避免底层连接异常时整个 A2A 请求无限挂起
MCP_TIMEOUT = int(os.getenv("TICKET_MCP_TIMEOUT", "45"))
# 单次 Agent 执行超时（秒）
AGENT_TIMEOUT = int(os.getenv("TICKET_AGENT_TIMEOUT", "100"))
# ==================== 票务查询函数 ====================
async def query_tickets(conversation: str) -> dict:
    """
    通过 LangChain Agent + MCP Tools 执行票务查询

    这个函数是 ticket_server 的核心查询逻辑。与 weather_server 的 Text-to-SQL 模式不同，
    这里使用的是"工具调用（Tool Calling）"模式：
    1. 从 MCP Server 加载所有可用的工具（火车票、机票、演唱会票等查询工具）
    2. 创建一个 LangChain Agent，让它自主选择调用哪个工具
    3. AgentExecutor 负责执行 Agent 循环：
       - LLM 分析用户输入，决定调用哪个工具
       - 执行工具调用
       - LLM 根据工具返回结果生成最终回复

    这种模式的优势：
    - 不需要手写 SQL，LLM 直接调用现成的工具
    - 支持多种票务类型（火车/飞机/演唱会），工具自动路由
    - 信息不足时，Agent 会自动追问用户

    参数：
        conversation (str): 用户的查询内容，例如：
            "火车票 北京 上海 2025-07-31 硬卧"
            "机票 北京 上海 2025-07-31 经济舱"

    返回值：
        dict: 查询结果，格式为：
            - {"status": "success", "message": "格式化后的查询结果"}  # 查询成功
            - {"status": "error", "message": "错误信息"}  # 查询失败
    """
    try:
        # 每次查询使用独立的 MCP Client：
        # handle_task 由 Flask 线程池并发调用，每个请求都有自己的事件循环，
        # 共享同一个 Client 会跨事件循环复用会话，导致请求相互卡死。
        async with Client(MCP_URL, timeout=MCP_TIMEOUT) as mcp_client:
        # 连接 MCP Server（端口 8001），建立通信通道
        # async with streamablehttp_client(MCP_URL) as (read, write, _):
        #     async with ClientSession(read, write) as session:
        #         await session.initialize()  # 初始化会话

                # 从 MCP 会话中加载所有可用工具
                # load_mcp_tools 会自动将 MCP Server 提供的工具转换为 LangChain 工具对象
                tools = await load_mcp_tools(mcp_client.session)

                # 定义 Agent 的系统 Prompt
                # 这个 Prompt 告诉 LLM：你是谁、你能做什么、输出格式要求
                prompt = ChatPromptTemplate.from_messages([
                    ("system", """你是一个票务查询助手，能够调用工具来完成火车票、飞机票或演唱会门票的查询。
你需要仔细分析用户的问题，从问题中提取工具需要的参数，然后调用对应的查询工具。
如果用户提供的信息不足以提取到调用工具所有必要参数，则向用户追问，以获取该信息。不能自己编撰参数。
每一轮只调用一个工具，等工具结果返回后再决定下一步，不要在同一轮并行调用多个工具。
查询到结果后，请用清晰的中文格式化输出，格式如下：
- 火车票：出发城市 到 到达城市 出发时间: 车次XX，座位类型XX，票价XX元，剩余XX张
- 机票：出发城市 到 到达城市 出发时间: 航班XX，舱位类型XX，票价XX元，剩余XX张
- 演唱会票：城市 开始时间: 艺人XX演唱会，场地XX，票价XX元，剩余XX张
如果未查到数据，请回复"未找到相关票务数据，请确认或修改查询条件。"
当前日期是{current_date}。"""),
                    ("human", "{input}"),  # 用户输入会替换到这里
                    ("placeholder", "{agent_scratchpad}"),  # Agent 思考过程的占位符
                ])

                # 创建基于工具调用的 Agent
                # create_tool_calling_agent 会创建一个能自主选择和使用工具的 Agent
                # LLM 以结构化格式（函数调用）来使用工具，而不是用自然语言描述
                agent = create_tool_calling_agent(llm, tools, prompt)

                # 创建 Agent 执行器
                # AgentExecutor 负责运行 Agent 循环：
                # 1. 将用户输入发给 LLM
                # 2. LLM 决定调用哪个工具 + 提供参数
                # 3. 执行工具调用
                # 4. 将工具结果返回给 LLM
                # 5. LLM 生成最终回复
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                # 获取当前日期，注入到 Prompt 中
                current_date = datetime.now(pytz.timezone('Asia/Shanghai')).strftime('%Y-%m-%d')

                # 执行 Agent，传入用户输入和当前日期
                # 加外层超时，避免 MCP/LLM 异常时请求永久挂起
                response = await asyncio.wait_for(
                    agent_executor.ainvoke({
                        "input": conversation,  # 用户输入的查询内容
                        "current_date": current_date  # 当前日期
                    }),
                    timeout=AGENT_TIMEOUT
                )

                # AgentExecutor 的 response["output"] 就是 LLM 生成的最终回复
                return {"status": "success", "message": response["output"]}

    except asyncio.TimeoutError:
        logger.error("票务查询超时")
        return {"status": "error", "message": "票务查询超时，请稍后重试。"}
    except ExceptionGroup as eg:
        # MCP 工具内部使用 anyio.create_task_group()，失败时抛出 ExceptionGroup
        first_exc = eg.exceptions[0] if eg.exceptions else eg
        logger.error(f"票务 MCP 查询出错：{first_exc}")
        return {"status": "error", "message": f"票务 MCP 查询出错：{first_exc}"}
    except BaseException as e:
        # 兜底捕获（包括 ExceptionGroup 等 BaseException 子类）
        logger.error(f"票务 MCP 查询出错：{str(e)}")
        return {"status": "error", "message": f"票务 MCP 查询出错：{str(e)}"}


# ==================== 票务预订函数 ====================
async def order_tickets(query: str) -> dict:
    """
    通过 LangChain Agent + MCP Tools 执行票务预订

    与查询函数类似，但用于票务预订场景。
    同样使用 Agent + 工具调用模式，LLM 会自动选择合适的预定工具。

    与查询的区别：
    - 预定需要更多的参数（如座位类型、数量等）
    - 预定操作会修改数据库状态，而查询只是读取数据

    参数：
        query (str): 用户的预订请求，例如：
            "北京 到 上海 2025-11-15 火车票 二等座 1张"

    返回值：
        dict: 预订结果，格式为：
            - {"status": "success", "message": "预订成功信息"}
            - {"status": "error", "message": "错误信息"}
    """
    try:
        # 连接 MCP Server
        async with streamablehttp_client(MCP_URL) as (read, write, _):
            async with ClientSession(read, write) as session:
                await session.initialize()  # 初始化会话

                # 加载 MCP 工具
                tools = await load_mcp_tools(session)

                # 定义预定 Agent 的系统 Prompt
                prompt = ChatPromptTemplate.from_messages([
                    ("system",
                     "你是一个票务预定助手，能够调用工具来完成火车票、飞机票或演出票的预定。你需要仔细分析工具需要的参数，然后从用户提供的信息中提取信息。如果用户提供的信息不足以提取到调用工具所有必要参数，则向用户追问，以获取该信息。不能自己编撰参数。"),
                    ("human", "{input}"),
                    ("placeholder", "{agent_scratchpad}"),
                ])

                # 创建预定 Agent
                agent = create_tool_calling_agent(llm, tools, prompt)
                agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

                # 执行 Agent
                response = await agent_executor.ainvoke({"input": query})

                return {"status": "success", "message": f"{response['output']}"}

    except Exception as e:
        logger.error(f"票务 MCP 预订出错：{str(e)}")
        return {"status": "error", "message": f"票务 MCP 预订出错：{str(e)}"}


# ==================== Agent Card（代理卡片） ====================
# 描述票务代理的能力、技能等信息
# 主助手通过这个卡片决定是否将票务相关任务路由到这个代理
agent_card = AgentCard(
    name="TicketAssistant",  # 代理名称
    description="基于 LangChain 提供票务查询和预订服务的统一助手",  # 代理描述
    url="http://127.0.0.1:5006",  # 代理的访问地址
    version="2.0.0",  # 版本号
    capabilities={"streaming": True, "memory": True},  # 支持的能力
    skills=[
        # 技能1：火车票查询
        AgentSkill(
            name="query train tickets",
            description="查询火车票/火车票，支持指定出发城市、到达城市、日期和座位类型",
            examples=["火车票 北京 上海 2025-07-31", "北京到广州的高铁 明天 二等座"]
        ),
        # 技能2：机票查询
        AgentSkill(
            name="query flight tickets",
            description="查询机票/航班，支持指定出发城市、到达城市、日期和舱位类型",
            examples=["机票 北京 上海 2025-07-31", "北京到深圳的机票 后天 经济舱"]
        ),
        # 技能3：演唱会门票查询
        AgentSkill(
            name="query concert tickets",
            description="查询演唱会门票，支持指定城市、艺人、日期和票档类型",
            examples=["演唱会 北京 刀郎 2025-08-23", "周杰伦演唱会门票 上海"]
        ),
        # 技能4：火车票预定
        AgentSkill(
            name="order train tickets",
            description="根据车次、座位类型和数量预定火车票",
            examples=["预定G1次列车 2025-11-15 北京到上海 二等座 1张"]
        ),
        # 技能5：机票预定
        AgentSkill(
            name="order flight tickets",
            description="根据航班号、舱位类型和数量预定机票",
            examples=["预定MU5101 2025-12-11 上海到北京 公务舱 2张"]
        ),
        # 技能6：演唱会门票预定
        AgentSkill(
            name="order concert tickets",
            description="根据艺人、场地、日期、票档和数量预定演唱会门票",
            examples=["预定刀郎演唱会 2025-08-23 北京 看台票 2张"]
        )
    ]
)


# ==================== 票务查询服务器类 ====================
class TicketQueryServer(A2AServer):
    """
    票务查询 A2A 服务器

    这个类继承自 A2AServer，负责处理所有票务相关的任务。
    它的工作流程比天气查询更简单，因为查询逻辑已经封装在 query_tickets 函数中
    （内部使用了 LangChain Agent + MCP Tools 模式）。

    任务处理流程：
    1. 接收来自主助手的任务（Task）
    2. 从任务中提取用户的查询内容
    3. 调用 query_tickets 函数执行查询
    4. 将查询结果设置到任务状态中，返回给主助手
    """

    def __init__(self):
        """
        初始化票务查询服务器
        """
        super().__init__(agent_card=agent_card)  # 调用父类初始化，注册 Agent Card

    def handle_task(self, task):
        """
        处理来自 A2A 客户端的任务 —— 本服务器的核心方法

        完整流程：
        1. 提取输入：从任务消息中获取用户的查询内容
        2. 调用 MCP 查询：通过 query_tickets 函数执行查询
           - query_tickets 内部使用 LangChain Agent + MCP Tools
           - LLM 自动选择合适的工具（火车票/机票/演唱会）
           - 工具调用后格式化结果返回
        3. 根据查询结果设置任务状态：
           - 成功：COMPLETED 状态，结果放在 task.artifacts
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
        # 从任务消息中获取内容
        content = (task.message or {}).get("content", {})
        # 提取对话内容，即客户端发起的任务中的用户输入
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        logger.info(f"用户查询: {conversation}")

        try:
            # ========== 步骤2：调用 MCP 查询 ==========
            # query_tickets 内部会：
            # 1. 连接 MCP Server 加载所有工具
            # 2. 创建 LangChain Agent
            # 3. Agent 自动选择合适的工具并执行
            # 4. 格式化结果返回
            query_result = asyncio.run(query_tickets(conversation))
            logger.info(f"MCP 查询返回: {query_result}")

            # ========== 步骤3：根据结果设置任务状态 ==========
            if query_result.get("status") == "success":
                # 查询成功：将结果文本放入任务产物（artifacts）
                result_text = query_result.get("message", "")
                task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                task.status = TaskStatus(state=TaskState.COMPLETED)

            elif query_result.get("status") == "error":
                # MCP 查询出错：设置为失败状态
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": query_result.get("message", "查询失败，请重试。")}}
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
                message={"role": "agent", "content": {"text": f"查询失败: {str(e)} 请重试或提供更多细节。"}}
            )
            return task


# ==================== 主函数：启动服务器 ====================
if __name__ == "__main__":
    # 创建票务查询服务器实例
    ticket_server = TicketQueryServer()

    # 打印服务器信息，方便确认启动状态
    print("\n=== 服务器信息 ===")
    print(f"名称: {ticket_server.agent_card.name}")
    print(f"描述: {ticket_server.agent_card.description}")
    print("\n技能:")
    for skill in ticket_server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")

    # 启动 A2A 服务器，监听 5006 端口
    # 主助手会通过 http://127.0.0.1:5006 连接本服务器
    run_server(ticket_server, host="0.0.0.0", port=5006)
