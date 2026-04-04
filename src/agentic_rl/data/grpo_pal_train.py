from __future__ import annotations

# 文件功能：
# 1) 把 `data/raw/gsm8k_train_pal.json` 转换为 GRPO 训练数据。
# 2) 仅保留 GYM 环境初始化需要的最小字段：messages/ground_truth/env_config。
# 3) 输出格式与 PAL 工具训练约定一致，供 ms-swift GRPO + GYM Env 直接消费。

import json
import re
from pathlib import Path

from agentic_rl.tools.python_tool import get_python_tool_spec_json


def _read_json(path: str | Path) -> list[dict]:
    """
    读取 JSON 数组文件。

    输入：
    - path: 源 JSON 文件路径。

    返回：
    - list[dict]: 样本列表。
    """

    with Path(path).open("r", encoding="utf-8") as file:
        data = json.load(file)
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
    - tuple[str, str]: (rationale, final_answer)
    """

    parts = raw_answer.split("####", maxsplit=1)
    rationale = parts[0].strip()
    final_answer = parts[1].strip() if len(parts) == 2 else raw_answer.strip()
    return rationale, final_answer


def _pick_final_answer(record: dict, fallback: str) -> str:
    """
    从增强字段中选择最终答案。

    输入：
    - record: PAL 原始样本。
    - fallback: 从 raw_answer 中解析得到的默认答案。

    返回：
    - str: 最终答案文本。
    """

    for key in ("extracted_answer", "answer_string"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback.strip()


def _normalize_ground_truth(value: str) -> str:
    """
    归一化 ground_truth 文本，便于奖励函数稳定比较。

    输入：
    - value: 原始答案文本。

    返回：
    - str: 归一化结果。
    """

    normalized = str(value).strip().replace(",", "").replace("$", "")
    if normalized.endswith("."):
        normalized = normalized[:-1].strip()
    # 重要逻辑：数值答案统一为最短十进制文本，减少字符串比较误差。
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", normalized):
        number = float(normalized)
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.12g}"
    return normalized


def _build_system_tools_message_content() -> str:
    """
    构建 system 消息中的 tools 字段文本。

    返回：
    - str: 形如 `"tools": [...]` 的字符串。
    """

    return f'"tools": {get_python_tool_spec_json()}'


def _build_grpo_record(record: dict) -> dict:
    """
    构建单条 GRPO 训练样本。

    结构：
    - messages: 仅包含 system/user 两条初始消息。
    - ground_truth: 最终标准答案。
    - env_config: 指定 GYM 环境名称与工具名。
    """

    question = str(record.get("question", "")).strip()
    raw_answer = str(record.get("raw_answer", "")).strip()
    _, fallback_final_answer = _split_answer(raw_answer)
    final_answer = _pick_final_answer(record, fallback_final_answer)

    return {
        "messages": [
            {"role": "system", "content": _build_system_tools_message_content()},
            {"role": "user", "content": question},
        ],
        "ground_truth": _normalize_ground_truth(final_answer),
        "env_config": {
            "name": "python_tool_env",
            "tool_name": "python",
        },
    }


def build_grpo_pal_train_dataset(
    train_raw_path: str | Path = "data/raw/gsm8k_train_pal.json",
    train_output_path: str | Path = "data/processed/grpo_pal_train.jsonl",
    train_records: list[dict] | None = None,
) -> dict[str, int]:
    """
    构建 PAL 的 GRPO 训练数据并落盘。

    输入：
    - train_raw_path: 原始 PAL 训练数据路径。
    - train_output_path: 输出 GRPO 训练 JSONL 路径。
    - train_records: 可选，直接传入已加载/已切分样本；传入时不再读取 train_raw_path。

    返回：
    - dict[str, int]: 包含 raw / processed 统计信息。
    """

    records = train_records
    if records is None:
        train_raw_path = Path(train_raw_path)
        if not train_raw_path.exists():
            raise FileNotFoundError(f"训练原始文件不存在: {train_raw_path}")
        records = _read_json(train_raw_path)

    processed_records = [_build_grpo_record(record) for record in records]
    processed_count = _write_jsonl(processed_records, train_output_path)
    return {"raw": len(records), "processed": processed_count}
