# 基于A2A框架的旅游智能体项目

## 环境

* python3.12 使用uv包管理或者pip
* 复制 `.env.example` 为 `.env` 并填入自己的密钥（`.env` 已被 `.gitignore` 忽略，请勿提交）
* 需要启动mysql 8.0.43服务和milvus 2.4.4数据库服务
* 数据导入在sql目录下，需要把数据写入mysql和milvus中。其中[init_tour_group_rag.py](sql/init_tour_group_rag.py)
  写入milvus的，其余是mysql的建表和数据导入脚本

## 智能体

| 智能体 | 端口 | 职责 |
| --- | ---: | --- |
| WeatherQueryAssistant | 5005 | 天气查询 |
| TicketAssistant | 5006 | 火车票、机票、演唱会票查询与预订 |
| TripAssistant | 5007 | 旅游团、租车、旅行保险查询与预订 |
| CopywritingAssistant | 5008 | 小红书旅行种草文案；会按需调用以上三个代理收集真实素材，并通过图片 MCP 检索景点实拍图 |

## 工具服务

| 服务 | 端口 | 说明 |
| --- | ---: | --- |
| 票务 MCP | 8001 | 火车票、机票、演唱会票 |
| 天气 MCP | 8002 | 和风天气 / 数据库天气 |
| 行程 MCP | 8003 | 旅游团、租车、保险 |
| 图片 MCP | 8004 | 景点实拍图检索，返回图片 URL、原图页与授权说明 |

图片 MCP 按 `IMAGE_PROVIDER_ORDER` 依次尝试图源，命中即返回。默认顺序为
`unsplash,pexels,pixabay,openverse,wikimedia,so360,bing`；有图库 Key 时优先使用正版图库，
能访问开源图库时可用 `openverse,wikimedia`，受限网络可用 `so360,bing`（当前 `.env` 即此配置）。
360/Bing 图片授权未知，输出会保留原图页并提示版权归原作者，仅用于课程演示。

## 启动服务

* 先mysql和milvus启动
* 启动[mcp_server](mcp_server)下面的服务（8001/8002/8003/8004）
* 启动[a2a_server](a2a_server)下面的服务（5005/5006/5007，最后启动 5008 文案服务）
* 启动主函数[api_server.py](api_server.py)（8085）

## 多智能体评估

所有服务启动后运行：

```bash
python tests/eval_multi_agent.py
```

评估集只有 7 条小样本（`tests/eval_multi_agent_dataset.json`），覆盖单代理查询、多意图协作、小红书文案的层级协作以及信息缺失追问。评估脚本会自动加载 `.env`，并在结束后还原评估前的短期记忆。

| 指标 | 说明 |
| --- | --- |
| Task Success Rate (TSR) | 关键词、格式、禁用词等约束全部满足的用例比例 |
| Routing Accuracy / Intent Micro-F1 | 意图识别与期望意图的一致性 |
| Agent Selection Accuracy | 路由到的顶层代理集合是否与期望一致 |
| Keyword Coverage | 答案对必需信息的覆盖程度 |
| Format Compliance | 小红书文案固定段落是否齐全（含【景点实拍】） |
| Factual Grounding | 文案是否引用了真实车次等事实数据 |
| Image Embedding Compliance | 是否输出可渲染的真实图片 Markdown |
| Error Rate | 超时、服务异常等错误比例 |
| Collaboration Breadth | 单个请求平均参与的顶层代理数 |
| Latency | 平均 / P95 / 最大端到端延迟 |

详细结果写入 `tests/eval_results.json`。常用参数：

```bash
python tests/eval_multi_agent.py --case xiaohongshu_note   # 只跑文案用例
python tests/eval_multi_agent.py --limit 3                 # 只跑前 3 条
```

## docker部署

已经创建了Dockerfile文件和docker-compose.yml文件，可以直接使用docker-compose启动服务
启动命令：docker-compose up -d，完成自动打包并且启动服务