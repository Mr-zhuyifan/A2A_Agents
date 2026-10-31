"""
需求：实现基于 A2A 的小红书旅行种草文案服务器

架构说明：
    本服务器是 SmartVoyage 系统中的文案子代理（CopywritingAssistant），
    运行在独立进程中（127.0.0.1:5008），通过 A2A 协议与主助手通信。

    它与其他子代理最大的区别是：它自己会作为 A2A 客户端，
    按需调用其他子代理收集真实素材。形成两层 Agent 协作：

        主助手 ChatService
             │ A2A
             ▼
        CopywritingAssistant (5008)
             ├─ TicketAssistant (5006) ──→ 票务 MCP (8001) ──→ MySQL
             ├─ TripAssistant   (5007) ──→ 行程 MCP (8003) ──→ MySQL / Milvus
             ├─ WeatherQueryAssistant (5005) ──→ 天气 MCP (8002) ──→ MySQL
             └─ 图片 MCP (8004) ──→ 图片检索（Unsplash/Pexels/Pixabay/Openverse/Wikimedia/360/Bing）

    这样可以避免文案代理直接访问数据库，保持"每个 Agent 只负责自己的领域"：
    - 交通车次、航班、演出票价 → 问票务代理
    - 景点路线、旅游团亮点、租车、保险 → 问行程管家
    - 目的地天气 → 问天气代理
    - 景点实拍图 → 问图片 MCP（返回结构化图片 URL、原图页和授权说明）
    - 文案代理负责整合事实、估算预算、撰写有真实体验感的种草笔记

    工具调用模式：
    LangChain Agent 将"调用其他 A2A Agent"封装成工具（Agent-as-Tool）。
    LLM 先决定需要哪些素材，再调用对应工具，最后根据工具返回的真实数据写文案。

    输出内容：
    - 3 个标题备选
    - 600~900 字第一人称小红书正文
    - 花费清单（交通等真实价格 + 住宿餐饮估算）
    - 配图建议与话题标签
    - 数据说明（哪些来自实时查询，哪些是估算）
"""

# ==================== 导入依赖 ====================
import asyncio  # 异步 IO
import json  # 解析图片 MCP 返回的 JSON
import os  # 环境变量
import re  # 景点名称抽取与段落替换
import uuid  # 任务 ID

from datetime import datetime  # 当前日期注入
import pytz  # 时区
from fastmcp import Client  # 图片 MCP 客户端

from python_a2a import (
    A2AServer, run_server, AgentCard, AgentSkill,
    TaskStatus, TaskState,
    TextContent, Message, MessageRole, Task,
)
from python_a2a.client import A2AClient

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.agents import create_tool_calling_agent, AgentExecutor

from config import Config
from create_logger import logger

conf = Config()

# ==================== 初始化大模型 ====================
llm = ChatOpenAI(**conf.llm_config)

# ==================== 下游 A2A 代理地址 ====================
# 文案代理需要向其他代理取素材，地址支持本地和 Docker 环境
AGENT_URLS = {
    "WeatherQueryAssistant": os.getenv("WEATHER_A2A_URL", "http://127.0.0.1:5005"),
    "TicketAssistant": os.getenv("TICKET_A2A_URL", "http://127.0.0.1:5006"),
    "TripAssistant": os.getenv("TRIP_A2A_URL", "http://127.0.0.1:5007"),
}

# 单个下游代理的超时时间（秒）
AGENT_TIMEOUT = int(os.getenv("COPYWRITING_AGENT_TIMEOUT", "180"))

# 景点实拍图 MCP（独立进程，端口 8004）
IMAGE_MCP_URL = os.getenv("IMAGE_MCP_URL", "http://127.0.0.1:8004/mcp")
IMAGE_TIMEOUT = int(os.getenv("IMAGE_MCP_TIMEOUT", "60"))
IMAGE_MAX_LANDMARKS = int(os.getenv("IMAGE_MAX_LANDMARKS", "4"))

# 下游代理（票务/行程/天气）内部只维护一个 MCP 客户端，
# 并发调用会互相干扰甚至卡死，因此文案代理串行地向它们取素材。
_DOWNSTREAM_LOCK = asyncio.Lock()


# ==================== 调用其他 A2A 代理 ====================
async def call_agent(agent_name: str, query: str) -> str:
    """
    通过 A2A 协议调用其他子代理，并把返回结果转成文本。

    参数：
        agent_name (str): 下游代理名称，必须是 AGENT_URLS 中的 key
        query (str): 交给下游代理的自然语言查询

    返回值：
        str: 下游代理返回的文本；调用失败时返回可读的错误说明，
             让文案 Agent 知道该素材缺失，而不是编造数据。
    """
    url = AGENT_URLS.get(agent_name)
    if not url:
        return f"未配置 {agent_name} 的地址，无法获取该数据。"

    client = A2AClient(url, timeout=AGENT_TIMEOUT)
    message = Message(content=TextContent(text=query), role=MessageRole.USER)
    task = Task(id="copy-" + str(uuid.uuid4()), message=message.to_dict())

    try:
        # 同一时刻只调用一个下游代理，避免多个请求争用下游的 MCP 客户端
        async with _DOWNSTREAM_LOCK:
            raw_response = await client.send_task_async(task)
        logger.info(f"文案代理调用 {agent_name} 返回: {raw_response}")

        if raw_response.status.state == "completed" and raw_response.artifacts:
            return raw_response.artifacts[0]["parts"][0]["text"]

        if raw_response.status.message:
            return raw_response.status.message.get("content", {}).get(
                "text", str(raw_response.status.message)
            )

        return f"{agent_name} 未返回有效数据。"

    except Exception as e:
        logger.error(f"文案代理调用 {agent_name} 失败: {e}")
        return (
            f"{agent_name} 暂时不可用（{e}）。"
            "请跳过该项数据，不要编造，并在【数据说明】中写明未获取到。"
        )


# ==================== 封装成 LangChain 工具 ====================
# 这些工具的 docstring 会作为工具描述交给 LLM，
# 所以必须写清楚：查什么、query 需要包含哪些参数、数据如何使用。

@tool
async def query_train_tickets(query: str) -> str:
    """查询火车票/高铁票的真实车次、出发与到达时间、座位类型、票价和余票。
    query 必须包含出发城市、到达城市、出发日期，可附加座位偏好，例如：
    “2026-11-01 深圳到成都，二等座，按价格从低到高”。
    返回的价格和车次是撰写文案的唯一依据，不得改写或编造。"""
    return await call_agent(
        "TicketAssistant",
        f"请查询以下火车票需求，并列出真实的车次、时间、座位类型、票价和余票：{query}\n"
        "注意：城市名保持原样（例如“深圳”“成都”），不要改写成车站名；"
        "“高铁/动车”不是座位类型，查询时不要把它当作 seat_type，"
        "请按二等座/一等座/商务座等座位类型查询，或直接不指定座位类型。",
    )


@tool
async def query_flight_tickets(query: str) -> str:
    """查询机票的真实航班号、起降时间、舱位、票价和余票。
    query 必须包含出发城市、到达城市、出发日期，可附加舱位偏好，例如：
    “2026-11-01 深圳到成都，经济舱，按价格从低到高”。
    返回的价格和航班号是撰写文案的唯一依据，不得改写或编造。"""
    return await call_agent(
        "TicketAssistant",
        f"请查询以下机票需求，并列出真实的航班号、起降时间、舱位、票价和余票：{query}\n"
        "注意：城市名保持原样（例如“深圳”“成都”），不要改写成机场名或“城市+机场”。",
    )


@tool
async def query_concert_tickets(query: str) -> str:
    """查询演唱会门票的真实场次、艺人、场馆、时间、票档和票价。
    query 需要包含城市和日期（用户提到艺人时也一并提供），例如：
    “2026-12-31 成都的跨年演唱会”。
    只有用户明确提到演唱会时才调用。"""
    return await call_agent(
        "TicketAssistant",
        f"请查询以下演唱会门票信息，并列出艺人、场馆、时间、票档和票价：{query}",
    )


@tool
async def query_tour_groups(query: str) -> str:
    """通过语义搜索查询目的地的旅游团线路，返回团名、天数、价格、亮点和出发日期。
    query 用自然语言描述需求即可，例如：“成都美食和文化深度游，适合年轻人”。
    用于获取景点、路线和玩法亮点，为文案中的攻略部分提供真实素材。"""
    return await call_agent(
        "TripAssistant",
        f"请查询以下旅游团/线路信息，并列出团名、天数、价格、亮点：{query}",
    )


@tool
async def query_car_rentals(query: str) -> str:
    """查询租车的真实车型、租车公司、日租价格和余量。
    query 需要包含取车城市、还车城市和日期，例如：
    “2026-11-01 成都取车，2026-11-04 成都还车”。
    只有用户提到自驾、租车或需要当地用车时才调用。"""
    return await call_agent(
        "TripAssistant",
        f"请查询以下租车信息，并列出车型、公司、日租价格和余量：{query}",
    )


@tool
async def query_insurance_products(query: str) -> str:
    """查询旅行保险的真实产品名称、价格、保障范围和保险公司。
    query 可包含保险类型或出行日期，例如：“含航班延误的旅行保险，2026-11-01”。
    只有用户提到保险或航班延误保障时才调用。"""
    return await call_agent(
        "TripAssistant",
        f"请查询以下旅行保险产品，并列出名称、价格、保障范围和保险公司：{query}",
    )


@tool
async def query_weather_forecast(query: str) -> str:
    """查询目的地未来天气预报，返回日期、天气现象、温度、湿度和风等。
    query 需要包含城市和日期/日期范围，例如：“2026-11-01 到 2026-11-04 成都天气”。
    只有在出行日期处于预报范围内，且文案需要写天气或穿搭提醒时才调用。
    查不到数据时不要编造天气。"""
    return await call_agent(
        "WeatherQueryAssistant",
        f"请查询以下天气信息，并返回日期、天气、温度和风力：{query}",
    )


TOOLS = [
    query_train_tickets,
    query_flight_tickets,
    query_concert_tickets,
    query_tour_groups,
    query_car_rentals,
    query_insurance_products,
    query_weather_forecast,
]


# ==================== 景点实拍图（图片 MCP） ====================
# 文案写完后，先让 LLM 从文案中提取真正写到的景点，再调用图片 MCP 检索实拍图，
# 最后确定性地把【景点实拍】段落插入文案。这样图片 URL 一定来自真实检索结果，
# 不依赖 LLM 复制长链接，也不会出现编造的图片地址。

CITY_NAMES = [
    "北京", "上海", "广州", "深圳", "成都", "重庆", "杭州", "南京", "武汉", "西安",
    "苏州", "厦门", "青岛", "长沙", "昆明", "大理", "丽江", "三亚", "哈尔滨", "天津",
    "郑州", "济南", "福州", "南宁", "贵阳", "拉萨", "兰州", "呼和浩特", "太原", "合肥",
    "南昌", "沈阳", "长春", "大连", "珠海", "桂林", "洛阳", "张家界", "黄山", "宁波",
    "无锡", "扬州", "温州", "泉州", "烟台", "敦煌", "九寨沟", "香格里拉", "西双版纳",
]

LANDMARK_EXTRACT_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """你是小红书旅游笔记的配图编辑。请从用户需求和已经生成的文案中，找出 3~4 个真正被写进文案的景点/地标，用来检索实拍照片。

规则：
- 只选具体的景点、地标、公园、街区、古镇、博物馆、寺院、山川湖海等地点；
- 不要选城市名、车站、机场、酒店、餐厅、商场、交通工具；
- 不要选“美食”“citywalk”“夜景”这类泛化词；
- 名称用大家熟悉的正式短名，最多 12 个字，必须能在文案中找到对应内容；
- 目的地城市填一个，找不到就填空字符串。

只输出 JSON，不要解释，格式：
{{"city": "成都", "landmarks": ["宽窄巷子", "锦里", "成都大熊猫繁育研究基地"]}}"""),
    ("human", "用户需求：\n{user_input}\n\n已生成文案：\n{copy_text}"),
])


def _parse_landmark_json(raw: str) -> tuple[str, list[str]]:
    """从 LLM 输出中解析 {"city": ..., "landmarks": [...]}，容忍 ```json 代码块。"""
    match = re.search(r"\{.*\}", raw or "", re.S)
    if not match:
        return "", []
    try:
        data = json.loads(match.group(0))
    except (ValueError, json.JSONDecodeError):
        return "", []

    city = str(data.get("city") or "").strip()
    if city not in CITY_NAMES:
        city = next((c for c in CITY_NAMES if c in city), "")

    landmarks = []
    for item in data.get("landmarks", [])[:6]:
        name = re.sub(r"[\[\]【】\"'“”‘’#*]", "", str(item)).strip()
        name = re.sub(r"^(?:去|到|在|逛|打卡|游览|漫步)", "", name).strip()
        if 2 <= len(name) <= 14 and name not in landmarks:
            landmarks.append(name)
    return city, landmarks


def _guess_city(user_input: str, copy_text: str) -> str:
    for city in CITY_NAMES:
        if city in user_input:
            return city
    match = re.search(r"(?:去|到|目的地[:：]?\s*)([一-龥]{2,8})", user_input)
    if match:
        candidate = match.group(1)
        if candidate in CITY_NAMES:
            return candidate
    for city in CITY_NAMES:
        if city in copy_text:
            return city
    return ""


def _fallback_landmarks(copy_text: str, city: str) -> list[str]:
    """LLM 抽取失败时，用“地点后缀”从正文里兜底找景点。"""
    pattern = re.compile(
        r"([一-龥]{2,12}(?:巷子|古镇|古城|古街|步行街|公园|博物馆|美术馆|"
        r"纪念馆|展览馆|寺|塔|宫|广场|草堂|祠|基地|剧院|书店|山|湖|桥|岛|里))"
    )
    prefixes = (
        "来到", "去了", "逛逛", "打卡", "走进", "漫步", "游览", "推荐", "然后",
        "我们", "这次", "顺便", "直接", "成都", "深圳", "在", "到", "去", "逛", "游",
    )
    names = []
    for match in pattern.finditer(copy_text):
        name = match.group(1)
        changed = True
        while changed:
            changed = False
            for prefix in prefixes:
                if name.startswith(prefix) and len(name) - len(prefix) >= 2:
                    name = name[len(prefix):]
                    changed = True
        if 2 <= len(name) <= 14 and name not in names:
            names.append(name)
    if not names and city:
        names = [city]
    return names[:IMAGE_MAX_LANDMARKS]


async def extract_landmarks(user_input: str, copy_text: str) -> tuple[str, list[str]]:
    """优先用 LLM 抽取景点，失败时退回正则，保证图片功能不会阻塞文案返回。"""
    try:
        response = await llm.ainvoke(LANDMARK_EXTRACT_PROMPT.format_messages(
            user_input=user_input[:2000],
            copy_text=copy_text[:6000],
        ))
        raw = response.content if hasattr(response, "content") else str(response)
        city, landmarks = _parse_landmark_json(str(raw))
        if landmarks:
            return city or _guess_city(user_input, copy_text), landmarks[:IMAGE_MAX_LANDMARKS]
    except Exception as exc:
        logger.warning(f"景点名称抽取失败，使用正则兜底：{exc}")

    city = _guess_city(user_input, copy_text)
    return city, _fallback_landmarks(copy_text, city)


def _safe_md_url(url: str) -> str:
    """只允许 http(s) 且不含空格/括号，避免破坏 Markdown 或注入属性。"""
    url = str(url or "").strip()
    if not re.match(r"^https?://", url, re.I):
        return ""
    if any(ch in url for ch in " \t\r\n\"'<>"):
        return ""
    return url.replace(")", "%29").replace("(", "%28")


def _safe_md_text(text: str) -> str:
    return re.sub(r"[\[\]*`_<>]", "", str(text or "")).strip()


async def _fetch_landmark_photo(client: Client, landmark: str, city: str) -> dict | None:
    result = await client.call_tool(
        "search_landmark_photos",
        {"landmark": landmark, "city": city, "count": 1},
        timeout=IMAGE_TIMEOUT,
    )
    data = getattr(result, "data", None)
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (ValueError, json.JSONDecodeError):
            data = None
    if not isinstance(data, dict):
        text = "".join(
            block.text for block in getattr(result, "content", []) if hasattr(block, "text")
        )
        try:
            data = json.loads(text)
        except (ValueError, json.JSONDecodeError):
            return None
    if data.get("status") != "success" or not data.get("images"):
        return None
    return data["images"][0]


def render_landmark_section(photos: list[tuple[str, dict]], error: str = "") -> str:
    """生成【景点实拍】正文（不含标题），图片和来源都来自图片 MCP 的真实返回。"""
    lines = []
    for index, (landmark, image) in enumerate(photos, start=1):
        image_url = _safe_md_url(image.get("image_url", ""))
        if not image_url:
            continue
        title = _safe_md_text(image.get("title") or landmark)[:80]
        site = _safe_md_text(image.get("site") or "")
        page_url = _safe_md_url(image.get("page_url", ""))
        license_name = _safe_md_text(image.get("license") or "")

        lines.append(f"{index}. **{_safe_md_text(landmark)}**")
        lines.append(f"![{_safe_md_text(landmark)}实拍]({image_url})")
        source = f"图源：{title}"
        if site and site not in title:
            source += f" · {site}"
        if page_url:
            source += f"｜[原图页]({page_url})"
        if license_name and license_name.lower() != "unknown":
            source += f"｜授权：{license_name}"
        else:
            source += "｜授权未知，商用需取得原作者授权"
        lines.append(source)
        lines.append("")

    if not lines:
        reason = "图片检索服务暂时不可用" if error else "暂未检索到可用的景点实拍图"
        lines.append(f"{reason}，可参考下方【配图建议】。")
    return "\n".join(lines).strip()


def _replace_section(text: str, heading: str, body: str) -> str:
    """替换同名段落内容；原段落缺失时插入到【配图建议】之前。"""
    pattern = re.compile(rf"{re.escape(heading)}.*?(?=\n【[^\n】]{{1,12}}】|\Z)", re.S)
    if pattern.search(text):
        return pattern.sub(f"{heading}\n{body}\n", text, count=1).rstrip() + "\n"

    for anchor in ("【配图建议】", "【标签】", "【数据说明】"):
        index = text.find(anchor)
        if index != -1:
            return (
                text[:index].rstrip()
                + f"\n\n{heading}\n{body}\n\n"
                + text[index:].lstrip()
            )
    return text.rstrip() + f"\n\n{heading}\n{body}\n"


def _append_image_note(text: str, note: str) -> str:
    if "【数据说明】" in text:
        return text.rstrip() + "\n" + note
    return text.rstrip() + "\n\n【数据说明】\n" + note


async def attach_landmark_photos(copy_text: str, user_input: str) -> str:
    """
    为已生成的文案补充真实景点实拍图，保留原有【配图建议】。

    流程：LLM 抽取景点 → 调图片 MCP（8004）逐张检索 → 确定性写入【景点实拍】。
    图片服务失败时仍然返回文案，并在段落与数据说明中写明未获取到。
    """
    city, landmarks = await extract_landmarks(user_input, copy_text)
    logger.info(f"文案配图：city={city}, landmarks={landmarks}")

    photos: list[tuple[str, dict]] = []
    error = ""
    if landmarks:
        try:
            async with Client(IMAGE_MCP_URL, timeout=IMAGE_TIMEOUT) as client:
                for landmark in landmarks[:IMAGE_MAX_LANDMARKS]:
                    try:
                        image = await _fetch_landmark_photo(client, landmark, city)
                    except Exception as exc:
                        error = str(exc)
                        logger.warning(f"检索“{landmark}”实拍图失败：{exc}")
                        continue
                    if image:
                        photos.append((landmark, image))
        except Exception as exc:
            error = str(exc)
            logger.warning(f"连接图片 MCP 失败：{exc}")

    section = render_landmark_section(photos, error=error)
    output = _replace_section(copy_text, "【景点实拍】", section)

    if photos:
        providers = sorted({image.get("provider") or "" for _, image in photos} - {""})
        source = "、".join(providers) or "图片检索"
        note = (
            f"- 景点实拍：共 {len(photos)} 张，来自 {source} 的真实图片检索；"
            "图片版权归原作者，仅用于课程演示，商用请通过原图页取得授权。"
        )
    else:
        note = "- 景点实拍：未检索到可用图片，已在【景点实拍】中说明。"
    return _append_image_note(output, note)


# ==================== 文案 Agent 系统提示 ====================
def build_prompt():
    """
    构建文案 Agent 的提示模板。

    LangChain 会把 {current_date} 替换成真实日期，
    {input} 是主助手转发过来的用户需求（可能包含对话历史）。
    """
    return ChatPromptTemplate.from_messages([
        ("system", """你是一位资深的小红书旅行内容策划与文案撰写专家，擅长把真实旅行经历写成有代入感、可收藏的种草笔记。读者是正在做旅行决策的年轻人。

【第一步：整理简报】
先从用户当前需求和对话历史中提取：出发地、目的地、出发日期、旅行天数、同行人、预算、兴趣偏好、必须提到的体验（演唱会、美食、景点等）。
- 缺少“目的地、出发地、旅行天数”中的任意一项时，不要编故事，只输出一句追问：
  需要补充信息：请告诉我出发城市、目的地和旅行天数。
  如果有多个缺失项，可以一次问清，但不要输出标题或正文。
- 如果只是缺少精确日期：可以选择当前日期之后最近的周六作为查询参考日，并在【数据说明】中注明；正文里用“周末/最近”等自然表达，不要虚构天气。
- 用户明确给出的信息优先，不能自行改动天数、出发地和预算。

【第二步：先查事实，再写文案】
在写正文之前，先按需调用工具获取真实数据，不要凭记忆编造价格、车次、航班或演出信息：
- 交通：需要写“从哪里出发、坐什么车/航班、多久、多少钱”时，调用 query_train_tickets 和/或 query_flight_tickets。
  · 用户没有指定交通方式时，火车和飞机都查，再按预算和体验择优写入。
  · 用户指定高铁/火车就查火车，指定飞机就查航班。
  · 如果查不到返程，就在花费清单中按“单程×2”估算并标注“往返估算”。
- 景点/玩法/攻略：调用 query_tour_groups 获取目的地的线路亮点，再结合你自身的旅行常识补充景点介绍和玩法；不要编造门票价格。
- 天气：出行日期在预报范围内且文案需要写天气/穿搭时，调用 query_weather_forecast；查不到就不写具体天气。
- 演唱会：只有用户提到时才调用 query_concert_tickets，并原样引用艺人、场馆、时间和票档价格。
- 租车/保险：只有用户提到自驾、家庭带娃或航班延误保障时才调用对应工具。
- 酒店、餐饮、市内交通没有工具时，可以给出合理估算区间，但必须写“约/人均/估算”，不能伪装成实时价格。

工具返回的数据是事实依据：航班号、车次、时间、票价必须原样使用，不得改写价格；数据缺失就省略或在【数据说明】里注明，绝不虚构。

【第三步：写一篇真实感的小红书笔记】
风格要求：
- 第一人称，像朋友分享真实体验，不像官方广告或说明书。
- 标题有吸引力但不标题党，3 个备选，每个 20 字以内，可带 1 个 emoji。
- 正文 600~900 字，短段落，适量 emoji（每段最多 2 个），可用【出发】【Day 1】【花费】这类小标题。
- 必须自然写到：从哪出发、怎么去（车次/航班 + 耗时 + 真实票价）、玩几天、去了哪些地方或体验了什么、住在什么区域/类型、吃了什么、总花费大约多少、2~4 条实用避坑建议。
- 不要出现“作为AI”“根据数据”“工具显示”“以下是”等字眼。
- 正文结尾用一句轻松的话引导收藏或评论。
- 正文里只写旅行体验和攻略，不要重复输出【花费清单】【景点实拍】【配图建议】【标签】这些小标题，这些统一放在后面的固定段落。
- 6~10 个小红书话题标签必须单独放在【标签】段落，不能写在【正文】末尾。

【预算规则】
- 交通和演出、租车、保险：使用工具查询到的真实价格。
- 住宿、餐饮、市内交通：给出合理估算，并标明“约”。
- 花费清单逐项相加，合计要算对；总花费按“每人”计算，并在标题旁注明人数假设（若用户未说明，按 2 人同行写）。
- 不确定的项目宁可标注“以实际为准”，也不要编造。

【输出格式】
必须严格按以下顺序输出 7 个段落标题，标题文字原样保留，一个都不能省略，
也不要自创或合并段落；不要添加额外解释：

【标题备选】
1. ...
2. ...
3. ...

【正文】
（完整的小红书笔记正文）

【花费清单】
- 往返交通：约 ¥xxx（真实车次/航班 + 价格，注明单程或往返）
- 住宿：约 ¥xxx（X 晚，估算）
- 餐饮：约 ¥xxx/人（估算）
- 门票/演出：¥xxx（真实数据；没有则写“未安排”）
- 当地交通/租车：约 ¥xxx（估算或真实数据）
- 保险：¥xxx（真实数据；没有则写“未购买”）
合计：约 ¥xxxx/人（2 人同行）

【景点实拍】
（这里由系统根据真实图片检索结果自动填充，你只需要原样保留这个标题，并写一行“由系统检索真实实拍图”，不要编造图片链接）

【配图建议】
1. ...
2. ...
3. ...

【标签】
#... #... #...

【数据说明】
- 交通/演出/租车/保险：来自实时查询（写查询日期）
- 住宿/餐饮/市内交通：为估算区间
- 未获取到的信息：...

当前日期：{current_date}。"""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])


# ==================== 文案生成核心逻辑 ====================
async def query_copywriting(conversation: str) -> dict:
    """
    执行一次小红书旅行种草文案生成。

    流程：
    1. 创建带工具的 LangChain Agent
    2. LLM 分析简报，按需调用下游 A2A 代理
    3. 拿到真实素材后生成完整笔记

    参数：
        conversation (str): 用户需求 + 由主助手拼接的对话历史

    返回值：
        dict: {"status": "success"/"error", "message": "..."}
    """
    try:
        current_date = datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d")

        agent = create_tool_calling_agent(llm, TOOLS, build_prompt())
        agent_executor = AgentExecutor(
            agent=agent,
            tools=TOOLS,
            verbose=True,
            max_iterations=18,  # 允许多轮工具调用：交通 + 景点 + 其他素材
        )

        response = await agent_executor.ainvoke({
            "input": conversation,
            "current_date": current_date,
        })

        output = (response.get("output") or "").strip()

        # 信息不足时直接追问，不做配图；正常文案则补真实景点实拍图
        if output and not output.startswith("需要补充信息"):
            try:
                output = await attach_landmark_photos(output, conversation)
            except Exception as exc:
                # 配图是增强能力，失败不能影响已经写好的文案
                logger.warning(f"文案配图失败，返回纯文案：{exc}")

        return {"status": "success", "message": output}

    except ExceptionGroup as eg:
        first_exc = eg.exceptions[0] if eg.exceptions else eg
        logger.error(f"文案生成出错：{first_exc}")
        return {"status": "error", "message": f"文案生成出错：{first_exc}"}
    except BaseException as e:
        logger.error(f"文案生成出错：{str(e)}")
        return {"status": "error", "message": f"文案生成出错：{str(e)}"}


# ==================== Agent Card ====================
agent_card = AgentCard(
    name="CopywritingAssistant",
    description="小红书旅行种草文案撰写助手，会按需调用票务、行程和天气代理收集真实素材，并通过图片 MCP 检索景点实拍图，生成个人体验风格的旅游营销文案",
    url="http://127.0.0.1:5008",
    version="1.0.0",
    capabilities={"streaming": True, "memory": True},
    skills=[
        AgentSkill(
            name="write xiaohongshu travel note",
            description="根据出发地、目的地、天数和偏好撰写小红书旅行种草笔记，包含标题、正文、花费清单、景点实拍图、配图建议和标签",
            examples=[
                "帮我写一篇深圳去成都4天的小红书种草文",
                "把我们这次成都跨年旅行写成小红书笔记",
            ],
        ),
        AgentSkill(
            name="write travel marketing copy",
            description="生成个人体验风格的旅游营销文案，可引用真实交通、演出、租车和保险数据",
            examples=[
                "写一篇成都美食之旅的种草文案，预算3000",
                "为我的成都周末游写一篇小红书笔记",
            ],
        ),
        AgentSkill(
            name="collect trip facts for copy",
            description="在写文案前自动调用票务、行程管家和天气代理收集真实素材，写完后通过图片 MCP 检索景点实拍图",
            examples=[
                "这篇文案需要真实的深圳到成都机票和演唱会票价",
                "查一下成都的旅游团亮点，写进攻略里",
            ],
        ),
    ],
)


# ==================== 文案 A2A 服务器 ====================
class CopywritingServer(A2AServer):
    """
    文案撰写 A2A 服务器。

    与普通子代理一样，接收主助手任务后调用 query_copywriting；
    不同之处是 query_copywriting 内部的工具会继续调用其他 A2A 代理。
    """

    def __init__(self):
        super().__init__(agent_card=agent_card)

    def handle_task(self, task):
        content = (task.message or {}).get("content", {})
        conversation = content.get("text", "") if isinstance(content, dict) else ""
        logger.info(f"文案需求: {conversation}")

        try:
            result = asyncio.run(query_copywriting(conversation))
            logger.info(f"文案生成返回: {str(result)[:300]}...")

            if result.get("status") == "success":
                result_text = result.get("message", "")

                # 关键信息不足时，文案 Agent 会以固定前缀返回追问
                if result_text.startswith("需要补充信息"):
                    task.status = TaskStatus(
                        state=TaskState.INPUT_REQUIRED,
                        message={"role": "agent", "content": {"text": result_text}},
                    )
                else:
                    task.artifacts = [{"parts": [{"type": "text", "text": result_text}]}]
                    task.status = TaskStatus(state=TaskState.COMPLETED)

            elif result.get("status") == "error":
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={
                        "role": "agent",
                        "content": {"text": result.get("message", "文案生成失败，请重试。")},
                    },
                )
            else:
                task.status = TaskStatus(
                    state=TaskState.FAILED,
                    message={"role": "agent", "content": {"text": "文案生成失败，请重试。"}},
                )

            return task

        except Exception as e:
            logger.error(f"文案生成失败: {str(e)}")
            task.status = TaskStatus(
                state=TaskState.FAILED,
                message={
                    "role": "agent",
                    "content": {"text": f"文案生成失败: {str(e)} 请重试或补充更多信息。"},
                },
            )
            return task


# ==================== 主函数 ====================
if __name__ == "__main__":
    copywriting_server = CopywritingServer()

    print("\n=== 服务器信息 ===")
    print(f"名称: {copywriting_server.agent_card.name}")
    print(f"描述: {copywriting_server.agent_card.description}")
    print("\n技能:")
    for skill in copywriting_server.agent_card.skills:
        print(f"- {skill.name}: {skill.description}")

    # 文案代理需要调用其他代理，打印下游地址方便排查
    print("\n下游代理:")
    for name, url in AGENT_URLS.items():
        print(f"- {name}: {url}")

    run_server(copywriting_server, host="0.0.0.0", port=5008)
