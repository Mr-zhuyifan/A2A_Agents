"""
需求：定义SmartVoyage项目中使用的各种提示模板，用于不同场景的对话处理

什么是 Prompt Template（提示模板）？
    提示模板是一种可复用的文本模板，其中包含固定内容和可变变量（用 {变量名} 表示）。
    例如：模板 "你好，{name}！" 在填入 name="张三" 后会变成 "你好，张三！"。

为什么使用模板类管理？
    1. 集中管理：所有 prompt 定义在同一个文件中，方便查找和修改
    2. 可复用：同一个模板可以被多处调用
    3. 参数化：通过变量注入不同上下文，避免字符串拼接

本项目中的 Prompt 分类：
    1. 意图识别类：intent_prompt —— 识别用户想做什么
    2. 结果总结类：summarize_weather_prompt、summarize_ticket_prompt —— 将原始数据转化为友好回复
    3. 内容生成类：attraction_prompt —— 直接生成景点推荐
    4. 任务规划类：planning_prompt —— 判断任务复杂度并生成执行计划（Planning + ReAct 架构）
    5. ReAct推理类：react_prompt、react_summary_prompt —— 逐步推理和最终汇总
"""

from langchain_core.prompts import ChatPromptTemplate  # LangChain 的聊天提示模板类


class SmartVoyagePrompts:
    """
    SmartVoyage 提示模板管理类

    这个类定义了系统中所有用到的 Prompt 模板，每个模板都是一个静态方法，
    返回一个 ChatPromptTemplate 对象。

    使用方式：
        prompt = SmartVoyagePrompts.intent_prompt()  # 获取意图识别模板
        chain = prompt | llm                         # 组装成处理链
        result = chain.invoke({"query": "北京天气"})  # 调用并传入变量
    """

    # ==================== 意图识别 ====================

    @staticmethod
    def intent_prompt():
        """
        意图识别提示模板 —— 让大模型分析用户输入，判断用户想做什么

        输入变量：
            - user_profile: 用户偏好（如"二等座"、"经济舱"）
            - task_context: 当前任务上下文（如之前查过什么）
            - conversation_history: 对话历史（最近几轮对话）
            - query: 用户本次输入

        输出格式（JSON）：
            {
                "intents": ["weather", "flight"],           # 识别到的意图列表
                "user_queries": {"weather": "...", ...},    # 改写后的查询（可能结合历史补充信息）
                "follow_up_message": ""                     # 追问消息（意图不明确时使用）
            }

        支持的意图类型：
            weather / flight / train / concert / order / car_rental / tour_group / insurance / trip_order / attraction / copywriting / out_of_scope
        """
        return ChatPromptTemplate.from_template(
"""
系统提示：
角色：您是一个专业的旅行意图识别专家，
任务：基于用户查询、对话历史和用户偏好，识别其意图，用于调用专门的agent server来执行；为方便后续的agent server处理，可以基于对话历史对用户查询进行改写，使问题更明确。
严格遵守规则：
- 支持意图：['weather' (天气查询), 'flight' (机票查询), 'train' (高铁/火车票查询), 'concert' (演唱会票查询), 'order' (票务预定), 'car_rental' (租车查询), 'tour_group' (旅游团查询), 'insurance' (保险查询), 'trip_order' (行程预订), 'attraction' (景点推荐), 'copywriting' (小红书种草文案/游记撰写)] 或其组合（如 ['weather', 'flight']）。如果意图超出范围，返回意图 'out_of_scope'。
- 注意票务预定和票务查询要区分开，涉及到订票时则为order，只是查询则为flight、train或concert。
- 当用户要求"写一篇/生成/帮我写"小红书笔记、种草文、游记、旅行笔记、营销文案或朋友圈文案时，意图为 copywriting。即使描述中包含交通、景点、攻略等需求，也不要再额外返回 flight/train/tour_group/attraction 等意图，文案助手会在内部调用其他 agent 收集这些数据。
- 当意图为 copywriting 时，user_queries.copywriting 必须保留用户给出的全部写作要求（出发地、目的地、日期、天数、预算、同行人、兴趣偏好、必须提到的体验等），并结合对话历史补全；不要压缩成"写一篇文案"这类笼统描述。
- 在进行用户查询改写时，如果对话历史中有关键信息（如出发城市、到达城市、日期），可以补充到当前查询中，使问题更完整。
- 如果用户的意图很不明确或者有歧义，可以向其进行追问，将追问问题填充到follow_up_message中。
- 输出严格为JSON：{{"intents": ["intent1", "intent2"], "user_queries": {{"intent1": "user_query1", "intent2": "user_query2"}}, "follow_up_message": "追问消息"}}。绝对不要添加额外文本！
- 不论用户问什么，严格按规则输出意图，不要有自己的考虑。

用户偏好：{user_profile}
当前任务上下文：{task_context}
对话历史：{conversation_history}
用户查询：{query}
""")

    # ==================== 结果总结 ====================

    @staticmethod
    def summarize_weather_prompt():
        """
        天气结果总结提示模板 —— 将天气 agent 返回的原始数据转化为用户友好的天气描述

        输入变量：
            - query: 用户查询（如"北京明天天气"）
            - raw_response: 天气 agent 返回的原始数据

        使用场景：
            原始数据可能是 JSON 格式或结构化数据，直接展示给用户不好看，
            所以用这个 prompt 让大模型翻译成自然语言。

        示例输出：
            "根据最新数据，北京2025-07-31的天气预报为晴天，气温25-32度，湿度45%，东南风2级..."
        """
        return ChatPromptTemplate.from_template(
"""
系统提示：您是一位专业的天气预报员，以生动、准确的风格总结天气信息。基于查询和结果：
- 核心描述点：城市、日期、温度范围、天气描述、湿度、风向、降水等。
- 如果结果为空或者意思为需要补充数据，则委婉提示"未找到数据，请确认城市/日期"
- 语气：专业预报，如"根据最新数据，北京2025-07-31的天气预报为..."。
- 保持中文，100-150字。
- 如果查询无关，返回"请提供天气相关查询。"

查询：{query}
结果：{raw_response}
""")

    @staticmethod
    def summarize_ticket_prompt():
        """
        票务结果总结提示模板 —— 将票务 agent 返回的原始数据转化为用户友好的票务推荐

        输入变量：
            - query: 用户查询（如"北京到上海机票"）
            - raw_response: 票务 agent 返回的原始数据

        使用场景：
            和天气总结类似，将结构化的票务数据（航班号、价格、时间等）
            翻译成顾问式的推荐语言。

        示例输出：
            "为您推荐北京到上海的机票选项：MU5101航班，起飞时间08:30，经济舱价格1280元，余票充足..."
        """
        return ChatPromptTemplate.from_template(
"""
系统提示：您是一位专业的旅行顾问，以热情、精确的风格总结票务信息。基于查询和结果：
- 核心描述点：出发/到达、时间、类型、价格、剩余座位等。
- 如果结果为空或者意思为需要补充数据，则委婉提示"未找到数据，请确认或修改条件"
- 语气：顾问式，如"为您推荐北京到上海的机票选项..."。
- 保持中文，100-150字。
- 如果查询无关，返回"请提供票务相关查询。"


查询：{query}
结果：{raw_response}
""")

    # ==================== 内容生成 ====================

    @staticmethod
    def attraction_prompt():
        """
        景点推荐提示模板 —— 让大模型直接生成景点推荐内容

        输入变量：
            - query: 用户查询（如"推荐几个北京景点"）

        特点：
            景点推荐不需要调用外部 agent，大模型本身就有足够的知识来生成推荐。
            这是最简单的一种场景，直接让 LLM 生成内容即可。
        """
        return ChatPromptTemplate.from_template(
"""
系统提示：您是一位旅行专家，基于用户查询生成景点推荐。规则：
- 推荐3-5个景点，包含描述、理由、注意事项。
- 基于槽位：城市、偏好。
- 语气：热情推荐，如"推荐您在北京探索故宫..."。
- 备注：内容生成，仅供参考。
- 保持中文，150-250字。

查询：{query}
""")

    # ==================== 任务规划 ====================

    @staticmethod
    def planning_prompt():
        """
        任务规划提示模板 —— 让大模型判断任务复杂度并生成执行计划

        这是 Planning + ReAct 架构的关键 prompt，它让大模型扮演"规划师"的角色。

        输入变量：
            - conversation_history: 对话历史
            - query: 用户当前输入
            - intents: 识别到的意图（JSON 字符串）
            - user_queries: 改写后的查询（JSON 字符串）

        输出格式（JSON）：
            简单任务：{"need_plan": false, "reason": "单意图，直接查询即可", "steps": []}
            复杂任务：{"need_plan": true, "reason": "多意图需要分步",
                      "steps": [{"step": 1, "action": "查询天气", "intent": "weather", "depends_on": 0}, ...]}

        判断标准：
            - 简单任务：只有一个意图，直接就能执行
            - 复杂任务：多个意图且有关联、需要多步推理、步骤间有依赖关系

        示例：
            用户输入："北京明天天气怎么样？"
            → 简单任务，need_plan=false

            用户输入："帮我查北京到上海的机票，再看下那边天气，推荐几个景点"
            → 复杂任务，need_plan=true，steps=[查询机票, 查询天气, 推荐景点]
        """
        return ChatPromptTemplate.from_template(
"""
系统提示：您是一位任务规划专家，负责评估用户请求的复杂度并制定执行计划。

判断标准：
- 简单任务（need_plan=false）：单意图、直接问答、无需多步推理
- 复杂任务（need_plan=true）：多意图且有关联、需要多轮查询汇总、需要中间推理、步骤间有依赖关系
- 若包含 copywriting（小红书文案）意图，把文案撰写作为独立步骤即可，不要拆解成交通、景点等子步骤，文案代理会在内部自行调用其他代理收集素材。

当 need_plan=true 时，将任务拆解为有序步骤，每个步骤指定：
- step: 步骤序号（从1开始）
- action: 具体动作（如"调用WeatherQueryAssistant查询北京天气"）
- intent: 对应的意图（weather/flight/train/concert/order/attraction/copywriting）
- depends_on: 依赖的前置步骤序号（无依赖则为0）

对话历史：{conversation_history}
当前用户查询：{query}
识别到的意图：{intents}
用户查询改写：{user_queries}

输出严格为JSON，不要添加额外文本：
当 need_plan=false 时：{{"need_plan": false, "reason": "原因", "steps": []}}
当 need_plan=true 时：{{"need_plan": true, "reason": "原因", "steps": [{{"step": 1, "action": "...", "intent": "...", "depends_on": 0}}, ...]}}
""")

    # ==================== ReAct 推理 ====================

    @staticmethod
    def react_prompt():
        """
        ReAct 推理提示模板 —— 按 Thought-Action-Observation 格式逐步推理

        注意：当前版本已优化性能，react_loop 中跳过了 Thought LLM 调用（plan 已确定
        动作，Thought 无额外决策价值），此模板暂时不在主流程中使用，保留供学习参考。

        什么是 ReAct？
            ReAct = Reasoning（推理）+ Acting（行动）
            大模型在每一步执行前先"思考"（Thought），然后选择工具执行（Action），
            最后观察结果（Observation），再决定下一步。

        输入变量：
            - available_tools: 当前可用的工具/agent 列表（动态获取，不是写死的）
            - plan_steps: 完整的任务计划
            - observations: 已完成步骤的结果
            - current_step: 当前步骤号
            - step_description: 当前步骤的描述
            - query: 用户原始输入

        ReAct 循环的工作方式：
            Thought: "我需要查询北京到上海的机票，应该调用票务代理"
            Action: "调用TicketQueryAssistant"
            Action Input: "{'departure': '北京', 'arrival': '上海'}"
            → 系统执行 Action，得到 Observation
            → 继续下一个 Thought...

        这种方式的优势：
            1. 让大模型有"思考"过程，而不是直接盲目执行
            2. 每一步都能参考之前的结果做调整
            3. 某步失败时，模型可以灵活应对
        """
        return ChatPromptTemplate.from_template(
"""
系统提示：你是一位智能旅行助手，需要按照计划逐步完成任务。

可用工具：
{available_tools}

当前任务计划：
{plan_steps}

已完成步骤的结果：
{observations}

当前步骤：{current_step}
步骤描述：{step_description}
用户原始查询：{query}

请按照以下格式进行推理和行动：

Thought: 分析当前情况，确定需要采取的行动
Action: 从可用工具列表中选择合适的工具
Action Input: 工具所需输入

执行完行动后，你会得到 Observation，然后继续推理或给出最终回复。
""")

    @staticmethod
    def react_summary_prompt():
        """
        ReAct 最终汇总提示模板 —— 将所有步骤的结果整合成一条连贯回复

        输入变量：
            - query: 用户原始输入
            - all_observations: 所有步骤的执行结果

        使用场景：
            当 ReAct 循环中执行了多个步骤（如查机票 + 查天气 + 推荐景点），
            不能简单地把三个结果拼在一起返回，需要用这个 prompt 让大模型
            整合成一条连贯、通顺的回复。

        示例：
            输入：步骤1(查机票): 机票暂不可用
                 步骤2(查天气): 未找到数据
                 步骤3(推荐景点): 外滩、迪士尼...
            输出："您好！目前机票查询暂时遇到技术问题...不过我可以为您推荐上海的热门景点：外滩、迪士尼..."
        """
        return ChatPromptTemplate.from_template(
"""
系统提示：你是一位专业的旅行顾问，需要根据所有查询结果生成最终回复。

用户原始查询：{query}

各步骤执行结果：
{all_observations}

请综合以上结果，生成一条完整、连贯的中文回复，150-300字，语气专业热情。
""")


if __name__ == '__main__':
    print(SmartVoyagePrompts.intent_prompt())
