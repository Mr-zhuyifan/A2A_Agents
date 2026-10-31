"""
SmartVoyage 多智能体评估脚本

评估目标：
    用一份很小的数据集，验证「意图识别 → 代理路由 → 多代理协作 → 最终答案」整条链路，
    并计算多智能体系统常用的评估指标。

评估指标：
    1. Task Success Rate (TSR)          任务成功率：关键词/格式/禁用词等约束全部满足
    2. Routing Accuracy                 路由准确率：识别出的意图集合与期望完全一致
    3. Intent Micro-F1                  意图级别 micro precision / recall / F1
    4. Agent Selection Accuracy         代理选择准确率：实际调用的代理集合与期望一致
    5. Keyword Coverage                 答案关键信息覆盖率
    6. Format Compliance                格式合规率（如小红书文案的 7 个固定段落）
    7. Factual Grounding                事实引用命中率（文案中是否引用了真实车次等）
    8. Image Embedding Compliance       实拍图渲染合规率（是否输出真实图片 Markdown）
    9. Error Rate                       错误率（服务异常/超时/明确失败）
   10. Collaboration Breadth            协作广度：单个请求平均路由到多少个顶层代理
   11. Latency                          平均 / P95 端到端延迟

运行前置条件：
    已启动 MCP 8001/8002/8003/8004、A2A 5005/5006/5007/5008，MySQL 中已导入演示数据。

运行方式：
    python tests/eval_multi_agent.py
    python tests/eval_multi_agent.py --limit 3
    python tests/eval_multi_agent.py --case xiaohongshu_note
    python tests/eval_multi_agent.py --output tests/eval_results.json

说明：
    - 脚本会加载项目根目录 .env（shell 中已有的环境变量优先），无需手动 export。
    - 评估会写入短期记忆表；脚本开始前快照、结束后自动还原，避免污染课堂演示上下文。
"""

import argparse
import asyncio
import json
import math
import os
import re
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

import pytz

# ==================== 路径与 .env ====================
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def load_dotenv(path: Path) -> None:
    """轻量加载 .env，shell 中已存在的变量优先。"""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


load_dotenv(ROOT / ".env")

import pymysql  # noqa: E402

from chat_service import ChatService, conf  # noqa: E402


DATASET_PATH = Path(__file__).with_name("eval_multi_agent_dataset.json")
DEFAULT_OUTPUT = Path(__file__).with_name("eval_results.json")
REQUIRED_PORTS = (8001, 8002, 8003, 8004, 5005, 5006, 5007, 5008)
ERROR_MARKERS = (
    "处理失败：", "意图识别JSON解析失败", "服务暂时不可用：",
    "文案生成出错", "Incorrect API key", "invalid_api_key",
)


# ==================== 带链路追踪的 ChatService ====================
class TracingChatService(ChatService):
    """记录意图识别结果和实际调用的代理，供评估使用。"""

    def __init__(self):
        super().__init__()
        self.detected_intents = []
        self.called_agents = []

        # 每个用例独立开始，避免评估用例之间通过记忆串场
        self.memory.short_term_messages = []
        self.memory.current_task = {}
        self.memory.entity_history = []
        self.messages = []
        self.conversation_history = ""

    def intent_agent(self, user_input):
        intents, user_queries, follow_up_message = super().intent_agent(user_input)
        self.detected_intents = list(intents)
        return intents, user_queries, follow_up_message

    async def _call_agent_intent(self, intent, query_str):
        agent_name = conf.intent.get(intent)
        if intent == "attraction":
            agent_name = "LLM"
        self.called_agents.append(agent_name or "unknown")
        return await super()._call_agent_intent(intent, query_str)


# ==================== 前置检查 ====================
def check_services() -> bool:
    """检查评估依赖的端口是否都在监听。"""
    down = []
    for port in REQUIRED_PORTS:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                down.append(port)
    if down:
        print("以下服务未启动，无法评估：" + ", ".join(str(p) for p in down))
        print("请先按 readme.md / 测试问题.md 启动 MCP 与 A2A 服务。")
        return False
    return True


# ==================== 记忆快照 ====================
def _db_connect():
    return pymysql.connect(
        host=conf.host,
        user=conf.user,
        password=conf.password,
        database=conf.database,
    )


def snapshot_short_term():
    """快照短期记忆，评估结束后还原。"""
    try:
        conn = _db_connect()
        cursor = conn.cursor()
        cursor.execute(
            "SELECT role, content, message_time, message_order "
            "FROM short_term_messages ORDER BY message_order ASC"
        )
        rows = cursor.fetchall()
        cursor.close()
        conn.close()
        return rows
    except Exception as exc:  # 数据库不可用时不影响评估本身
        print(f"[warn] 短期记忆快照失败：{exc}")
        return None


def restore_short_term(rows) -> None:
    if rows is None:
        return
    try:
        conn = _db_connect()
        cursor = conn.cursor()
        cursor.execute("DELETE FROM short_term_messages")
        for role, content, message_time, message_order in rows:
            cursor.execute(
                "INSERT INTO short_term_messages (role, content, message_time, message_order) "
                "VALUES (%s, %s, %s, %s)",
                (role, content, message_time, message_order),
            )
        conn.commit()
        cursor.close()
        conn.close()
        print("\n[info] 已还原评估前的短期记忆。")
    except Exception as exc:
        print(f"\n[warn] 短期记忆还原失败：{exc}")


# ==================== 评分 ====================
def _missing_required(case, answer):
    missing = [k for k in case.get("required_keywords", []) if k not in answer]
    for group in case.get("required_keyword_groups", []):
        if not any(keyword in answer for keyword in group):
            missing.append("任一：" + " / ".join(group))
    return missing


def evaluate_case(case, answer, actual_intents, actual_agents, latency, error):
    expected_intents = set(case.get("expected_intents", []))
    expected_agents = set(case.get("expected_agents", []))
    actual_intent_set = set(actual_intents)
    actual_agent_set = set(actual_agents)

    missing = _missing_required(case, answer)
    forbidden_hits = [k for k in case.get("forbidden_keywords", []) if k in answer]
    missing_sections = [s for s in case.get("required_sections", []) if s not in answer]
    missing_patterns = [
        pattern for pattern in case.get("required_patterns", [])
        if not re.search(pattern, answer, re.S)
    ]

    keyword_total = len(case.get("required_keywords", [])) + len(
        case.get("required_keyword_groups", [])
    )
    keyword_coverage = 1.0 if keyword_total == 0 else (keyword_total - len(missing)) / keyword_total

    grounding_ok = None
    grounding_groups = case.get("grounding_keyword_groups")
    if grounding_groups:
        grounding_ok = all(
            any(keyword in answer for keyword in group) for group in grounding_groups
        )

    format_ok = not missing_sections
    pattern_ok = not missing_patterns
    constraints_ok = not missing and not forbidden_hits
    success = constraints_ok and format_ok and pattern_ok and not error

    return {
        "id": case["id"],
        "category": case.get("category", ""),
        "query": case["query"],
        "answer": answer,
        "expected_intents": sorted(expected_intents),
        "actual_intents": sorted(actual_intent_set),
        "routing_exact": actual_intent_set == expected_intents,
        "expected_agents": sorted(expected_agents),
        "actual_agents": sorted(a for a in actual_agent_set if a),
        "agent_exact": actual_agent_set == expected_agents,
        "missing_keywords": missing,
        "forbidden_hits": forbidden_hits,
        "required_sections": case.get("required_sections", []),
        "missing_sections": missing_sections,
        "required_patterns": case.get("required_patterns", []),
        "missing_patterns": missing_patterns,
        "keyword_coverage": round(keyword_coverage, 4),
        "format_ok": format_ok,
        "pattern_ok": pattern_ok,
        "grounding_ok": grounding_ok,
        "success": success,
        "error": error,
        "collaboration_breadth": len({a for a in actual_agent_set if a}),
        "latency_s": round(latency, 2),
    }


# ==================== 汇总指标 ====================
def _micro_prf(records, expected_key, actual_key):
    tp = fp = fn = 0
    for record in records:
        expected = set(record[expected_key])
        actual = set(record[actual_key])
        tp += len(expected & actual)
        fp += len(actual - expected)
        fn += len(expected - actual)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
    }


def _p95(values):
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(0.95 * len(ordered)) - 1)
    return ordered[index]


def summarize(results):
    total = len(results)
    latencies = [r["latency_s"] for r in results]
    format_cases = [r for r in results if r["required_sections"]]
    grounding_cases = [r for r in results if r["grounding_ok"] is not None]
    pattern_cases = [r for r in results if r["required_patterns"]]

    return {
        "sample_count": total,
        "task_success_rate": round(sum(r["success"] for r in results) / total, 4),
        "routing_accuracy": round(sum(r["routing_exact"] for r in results) / total, 4),
        "intent_micro": _micro_prf(results, "expected_intents", "actual_intents"),
        "agent_selection_accuracy": round(sum(r["agent_exact"] for r in results) / total, 4),
        "keyword_coverage": round(
            sum(r["keyword_coverage"] for r in results) / total, 4
        ),
        "format_compliance": (
            round(sum(r["format_ok"] for r in format_cases) / len(format_cases), 4)
            if format_cases
            else None
        ),
        "factual_grounding": (
            round(
                sum(bool(r["grounding_ok"]) for r in grounding_cases) / len(grounding_cases),
                4,
            )
            if grounding_cases
            else None
        ),
        "image_embedding_compliance": (
            round(sum(r["pattern_ok"] for r in pattern_cases) / len(pattern_cases), 4)
            if pattern_cases
            else None
        ),
        "error_rate": round(sum(bool(r["error"]) for r in results) / total, 4),
        "avg_collaboration_breadth": round(
            sum(r["collaboration_breadth"] for r in results) / total, 4
        ),
        "latency_avg_s": round(sum(latencies) / total, 2),
        "latency_p95_s": round(_p95(latencies), 2),
        "latency_max_s": round(max(latencies), 2),
    }


# ==================== 执行 ====================
async def run_case(case, timeout):
    service = TracingChatService()
    started = time.perf_counter()
    error = None
    answer = ""

    try:
        answer = await asyncio.wait_for(service.chat(case["query"]), timeout=timeout)
    except asyncio.TimeoutError:
        error = f"timeout>{timeout}s"
        answer = ""
    except Exception as exc:  # ChatService 通常内部吞异常，这里兜底
        error = f"{type(exc).__name__}: {exc}"

    latency = time.perf_counter() - started

    if not error:
        for marker in ERROR_MARKERS:
            if marker in answer:
                error = marker
                break
        if not answer.strip():
            error = "empty answer"

    return evaluate_case(
        case=case,
        answer=answer,
        actual_intents=service.detected_intents,
        actual_agents=service.called_agents,
        latency=latency,
        error=error,
    )


def print_case_result(index, total, result):
    flag = "PASS" if result["success"] else "FAIL"
    intent_flag = "OK" if result["routing_exact"] else "MISMATCH"
    agent_flag = "OK" if result["agent_exact"] else "MISMATCH"
    print(
        f"[{index}/{total}] {result['id']:<30} {flag:<4} "
        f"{result['latency_s']:>7.1f}s  intents={intent_flag:<8} agents={agent_flag:<8} "
        f"coverage={result['keyword_coverage']:.0%}"
    )
    if result["actual_intents"] != result["expected_intents"]:
        print(f"        意图：期望 {result['expected_intents']} / 实际 {result['actual_intents']}")
    if result["actual_agents"] != result["expected_agents"]:
        print(f"        代理：期望 {result['expected_agents']} / 实际 {result['actual_agents']}")
    if result["missing_keywords"]:
        print(f"        缺少：{result['missing_keywords']}")
    if result["missing_sections"]:
        print(f"        缺少段落：{result['missing_sections']}")
    if result["missing_patterns"]:
        print(f"        缺少匹配：{result['missing_patterns']}")
    if result["forbidden_hits"]:
        print(f"        命中禁用词：{result['forbidden_hits']}")
    if result["error"]:
        print(f"        错误：{result['error']}")


def print_summary(summary):
    print("\n" + "=" * 64)
    print("多智能体评估结果")
    print("=" * 64)
    print(f"样本数              : {summary['sample_count']}")
    print(f"任务成功率 (TSR)    : {summary['task_success_rate']:.2%}")
    print(f"路由准确率          : {summary['routing_accuracy']:.2%}")
    print(
        "意图 Micro-F1       : "
        f"F1={summary['intent_micro']['f1']:.2%} "
        f"(P={summary['intent_micro']['precision']:.2%}, "
        f"R={summary['intent_micro']['recall']:.2%})"
    )
    print(f"代理选择准确率      : {summary['agent_selection_accuracy']:.2%}")
    print(f"关键词覆盖率        : {summary['keyword_coverage']:.2%}")
    if summary["format_compliance"] is not None:
        print(f"格式合规率          : {summary['format_compliance']:.2%}")
    if summary["factual_grounding"] is not None:
        print(f"事实引用命中率      : {summary['factual_grounding']:.2%}")
    if summary["image_embedding_compliance"] is not None:
        print(f"实拍图渲染合规率    : {summary['image_embedding_compliance']:.2%}")
    print(f"错误率              : {summary['error_rate']:.2%}")
    print(f"顶层代理参与度      : {summary['avg_collaboration_breadth']:.2f} 个代理/请求")
    print(
        f"延迟                : 平均 {summary['latency_avg_s']:.1f}s"
        f" | P95 {summary['latency_p95_s']:.1f}s"
        f" | 最大 {summary['latency_max_s']:.1f}s"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="SmartVoyage 多智能体评估")
    parser.add_argument("--case", action="append", default=None, help="只跑指定用例，可重复")
    parser.add_argument("--limit", type=int, default=None, help="最多跑多少条用例")
    parser.add_argument("--timeout", type=float, default=420, help="单条用例超时（秒）")
    parser.add_argument("--dataset", type=Path, default=DATASET_PATH, help="评估集 JSON 路径")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="结果 JSON 输出路径")
    parser.add_argument("--no-restore-memory", action="store_true", help="不还原评估前的短期记忆")
    return parser.parse_args()


async def main():
    args = parse_args()

    if not check_services():
        return 1

    dataset = json.loads(args.dataset.read_text(encoding="utf-8"))
    cases = dataset.get("cases", [])
    if args.case:
        selected = set(args.case)
        cases = [c for c in cases if c["id"] in selected]
        missing = selected - {c["id"] for c in cases}
        if missing:
            print(f"未找到用例：{', '.join(sorted(missing))}")
            return 1
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print("评估集为空。")
        return 1

    snapshot = None if args.no_restore_memory else snapshot_short_term()
    print(f"加载评估集 {args.dataset.name}，共 {len(cases)} 条用例")
    print(f"模型：{os.getenv('LLM_MODEL', '(config 默认值)')}，单条超时 {args.timeout:.0f}s\n")

    results = []
    try:
        for index, case in enumerate(cases, start=1):
            result = await run_case(case, args.timeout)
            results.append(result)
            print_case_result(index, len(cases), result)
    finally:
        if not args.no_restore_memory:
            restore_short_term(snapshot)

    summary = summarize(results)
    print_summary(summary)

    report = {
        "timestamp": datetime.now(pytz.timezone("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(args.dataset),
        "llm_model": os.getenv("LLM_MODEL", ""),
        "summary": summary,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"\n详细结果已写入：{args.output}")

    return 0 if summary["task_success_rate"] >= 0.8 else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
