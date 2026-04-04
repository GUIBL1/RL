from __future__ import annotations

# 文件功能：
# 1) 把 `data/raw/gsm8k_train_pal.json` 转换为 PAL SFT 训练数据。
# 2) 清洗后统一输出 `messages` 五段结构：system/user/assistant/user/assistant。
# 3) 第一段 assistant 采用 Thought + Use Tool + Tool Call，第四段 user 放置工具返回结果。

import json
import re
from pathlib import Path

from agentic_rl.tools.python_tool import (
    get_python_tool_spec_json,
    run_python_tool_from_tool_call_content,
)

EQUATION_PATTERN = re.compile(r"<<\s*([^=<>]+?)\s*=\s*([^<>]+?)\s*>>")

def _read_json(path: str | Path) -> list[dict]:
    """
    读取 JSON 数组文件。

    输入：
    - path: 源 JSON 文件路径。

    返回：
    - list[dict]: 样本列表。
    """

    # 读取原始训练数据。
    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
    # 保证输入格式稳定为数组。
    if not isinstance(data, list):
        raise ValueError(f"JSON 文件不是数组: {path}")
    return data


def _write_jsonl(records: list[dict], output_path: str | Path) -> int:
    """
    把样本列表写入 JSONL。

    输入：
    - records: 待写入记录。
    - output_path: 输出 JSONL 路径。

    返回：
    - int: 实际写入条数。
    """

    output_path = Path(output_path)
    # 自动创建输出目录，避免首次运行失败。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _split_answer(raw_answer: str) -> tuple[str, str]:
    """
    拆分 GSM8K 标准答案字段。

    输入：
    - raw_answer: 形如“推理过程 + #### 最终答案”的文本。

    返回：
    - (rationale, final_answer)
    """

    # `####` 后通常是最终答案。
    parts = raw_answer.split("####", maxsplit=1)
    rationale = parts[0].strip()
    final_answer = parts[1].strip() if len(parts) == 2 else raw_answer.strip()
    return rationale, final_answer


def _build_thought(record: dict, rationale: str) -> str:
    """
    生成训练样本中的 Thought 文本（尽量保留完整原文）。

    输入：
    - record: PAL 原始样本。
    - rationale: `raw_answer` 在 `####` 前的推理文本。

    返回：
    - str: 完整 Thought 内容。
    """

    chains = record.get("chains")
    if isinstance(chains, list) and chains:
        # 保留 chain 原文顺序和细节，尽量保持原始推理痕迹。
        thought = "\n".join(str(item).strip() for item in chains if str(item).strip())
        if thought:
            return thought
    # chains 缺失时使用原始 rationale。
    return rationale.strip()


def _extract_code(record: dict) -> str:
    """
    从 PAL 增强样本中提取可执行代码片段。

    输入：
    - record: PAL 原始样本。

    返回：
    - str: 代码文本，提取失败时为空串。
    """

    code_field = record.get("code")
    if isinstance(code_field, list) and code_field:
        first_block = code_field[0]
        if isinstance(first_block, list):
            # `code` 可能按行分片，先拼接回完整代码。
            code = "\n".join(str(line) for line in first_block if str(line).strip())
            if code.strip():
                return code.strip()
        if isinstance(first_block, str) and first_block.strip():
            return first_block.strip()

    generation_field = record.get("generation")
    if isinstance(generation_field, list) and generation_field:
        first_generation = generation_field[0]
        if isinstance(first_generation, list) and first_generation:
            # `generation` 中常见二维列表，取第一个候选。
            candidate = str(first_generation[0]).strip()
            if candidate:
                return candidate
        if isinstance(first_generation, str) and first_generation.strip():
            return first_generation.strip()

    return ""


def _extract_equations(rationale: str) -> list[str]:
    """
    从推理文本中提取 `<<expr=result>>` 的 expr 部分。

    输入：
    - rationale: 原始推理文本。

    返回：
    - list[str]: 可直接执行的表达式序列。
    """

    equations: list[str] = []
    for match in EQUATION_PATTERN.finditer(rationale):
        # 统一去掉千分位逗号，避免表达式执行报错。
        expr = match.group(1).replace(",", "").strip()
        if expr:
            equations.append(expr)
    return equations


def _build_fallback_code(rationale: str, final_answer: str) -> str:
    """
    当样本缺少代码时，基于推理文本/答案构造兜底代码。

    输入：
    - rationale: 推理文本。
    - final_answer: 最终答案。

    返回：
    - str: 可执行 Python 代码（包含 print）。
    """

    equations = _extract_equations(rationale)
    if equations:
        # 将多步表达式显式展开为 step_1, step_2 ...
        lines = [f"step_{index} = {expr}" for index, expr in enumerate(equations, start=1)]
        lines.append(f"print(step_{len(equations)})")
        return "\n".join(lines)

    normalized = final_answer.replace(",", "").strip()
    # 数值答案可直接赋值计算。
    if re.fullmatch(r"-?\d+(\.\d+)?", normalized):
        return f"result = {normalized}\nprint(result)"
    # 文本答案走字符串分支，避免语法错误。
    return f"result = {json.dumps(final_answer, ensure_ascii=False)}\nprint(result)"


def _normalize_code(code: str, fallback_code: str) -> str:
    """
    规范化要执行的 Action Input 代码。

    输入：
    - code: 从样本提取的候选代码。
    - fallback_code: 兜底代码。

    返回：
    - str: 最终 Action Input。
    """

    candidate = code.strip() if code else ""
    if not candidate:
        return fallback_code
    # PAL 样本常见 `def solution()`，补齐打印调用以产生可观测输出。
    if "def solution" in candidate and "print(solution())" not in candidate:
        return f"{candidate}\n\nprint(solution())"
    return candidate


def _pick_final_answer(record: dict, fallback: str) -> str:
    """
    从增强字段中选择最终答案。

    输入：
    - record: PAL 原始样本。
    - fallback: 从 `raw_answer` 解析得到的默认答案。

    返回：
    - str: 最终答案文本。
    """

    for key in ("extracted_answer", "answer_string"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback.strip()


def _build_system_tools_message_content() -> str:
    """
    构建 system 消息中的 tools 字段文本。

    返回：
    - str: 形如 `"tools": [...]` 的字符串。
    """

    # 直接复用工具规范，确保训练数据与执行协议严格一致。
    return f'"tools": {get_python_tool_spec_json()}'


def _build_tool_call_content(code: str) -> str:
    """
    构建 python 工具调用 JSON 字符串。

    输入：
    - code: 需要执行的 Python 代码。

    返回：
    - str: `{"name":"python","arguments":{"code":"..."}}`。
    """

    return json.dumps(
        {"name": "python", "arguments": {"code": code}},
        ensure_ascii=False,
    )


def _build_assistant_tool_use_content(thought: str, tool_call: str) -> str:
    """
    构建第一段 assistant 消息内容。

    输入：
    - thought: 推理过程文本。
    - tool_call: 工具调用 JSON 字符串。

    返回：
    - str: Thought + Use Tool + Tool Call 的组合文本。
    """

    # 重要逻辑：显式固定为 Use Tool + Tool Call，和评测解析逻辑保持同构。
    return (
        f"Thought: {thought}\n\n"
        "Use Tool: python\n"
        "Tool Call:\n"
        f"{tool_call}"
    )


def _build_sft_record(record: dict) -> dict:
    """
        构建单条 PAL SFT 训练样本。

    结构：
    - messages:
            1) system("tools": [...])
            2) user(问题)
            3) assistant(Thought + Use Tool + Tool Call)
            4) user(工具返回 JSON)
            5) assistant(Final Answer)
    """

    question = str(record.get("question", "")).strip()
    raw_answer = str(record.get("raw_answer", "")).strip()
    rationale, fallback_final_answer = _split_answer(raw_answer)
    final_answer = _pick_final_answer(record, fallback_final_answer)

    # 保留完整 Thought（不做内容压缩）。
    thought = _build_thought(record, rationale)
    if not thought:
        thought = "使用 python 工具计算并验证结果。"

    raw_code = _extract_code(record)
    fallback_code = _build_fallback_code(rationale, final_answer)
    action_code = _normalize_code(raw_code, fallback_code)

    # 重要逻辑：tool_call 的 code 与 assistant 中 Tool Call 字段必须完全一致。
    tool_call = _build_tool_call_content(action_code)
    assistant_react_content = _build_assistant_tool_use_content(thought, tool_call)

    # 直接调用工具实现产出返回，确保清洗协议与运行协议一致。
    tool_response = run_python_tool_from_tool_call_content(tool_call)
    system_tools_content = _build_system_tools_message_content()

    return {
        "messages": [
            {"role": "system", "content": system_tools_content},
            {"role": "user", "content": question},
            {"role": "assistant", "content": assistant_react_content},
            {"role": "user", "content": tool_response},
            {"role": "assistant", "content": f"Final Answer: {final_answer}"},
        ]
    }


def build_sft_pal_train_dataset(
    train_raw_path: str | Path = "data/raw/gsm8k_train_pal.json",
    train_output_path: str | Path = "data/processed/sft_pal_train.jsonl",
    train_records: list[dict] | None = None,
) -> dict[str, int]:
    """
    构建 PAL 训练数据并落盘。

    输入：
    - train_raw_path: 原始 PAL 训练数据路径。
    - train_output_path: 输出 SFT 训练 JSONL 路径。
    - train_records: 可选，直接传入已加载/已切分样本；传入时不再读取 train_raw_path。

    返回：
    - dict: 包含 raw / processed 统计信息。
    """

    records = train_records
    if records is None:
        train_raw_path = Path(train_raw_path)
        # 显式校验输入文件，便于快速定位路径错误。
        if not train_raw_path.exists():
            raise FileNotFoundError(f"训练原始文件不存在: {train_raw_path}")
        records = _read_json(train_raw_path)

    # 全量转换并写出结果。
    processed_records = [_build_sft_record(record) for record in records]
    processed_count = _write_jsonl(processed_records, train_output_path)
    return {"raw": len(records), "processed": processed_count}