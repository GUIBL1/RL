from __future__ import annotations

# 文件功能：
# 1) 把 `data/raw/gsm8k_test.jsonl` 转换为评测使用的简化问答格式。
# 2) 从原始 `answer` 中提取 `####` 后的最终答案。

import json
from pathlib import Path


def _read_jsonl(path: str | Path) -> list[dict]:
    """
    读取 JSONL 文件为字典列表。

    输入：
    - path: JSONL 文件路径。

    返回：
    - list[dict]: 逐行解析后的记录。
    """

    records: list[dict] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            # 跳过空行，避免 JSON 解析异常。
            if not line:
                continue
            records.append(json.loads(line))
    return records


def _write_jsonl(records: list[dict], output_path: str | Path) -> int:
    """
    把记录写入 JSONL 文件。

    输入：
    - records: 待写入记录。
    - output_path: 输出路径。

    返回：
    - int: 写入条数。
    """

    output_path = Path(output_path)
    # 自动创建输出目录。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _split_answer(raw_answer: str) -> tuple[str, str]:
    """
    拆分 GSM8K 的 answer 文本。

    输入：
    - raw_answer: 原始答案文本。

    返回：
    - tuple[str, str]: (原始文本, 最终答案)
    """

    # 约定 `####` 后是最终可评测答案。
    parts = raw_answer.split("####", maxsplit=1)
    final_answer = parts[1].strip() if len(parts) == 2 else raw_answer.strip()
    return raw_answer.strip(), final_answer


def _build_test_record(record: dict) -> dict:
    """
    构建单条测试样本。

    输入：
    - record: 原始 gsm8k 测试样本。

    返回：
    - dict: {question, answer}
    """

    question = str(record.get("question", "")).strip()
    _, answer = _split_answer(str(record.get("answer", "")).strip())
    return {"question": question, "answer": answer}


def build_pal_test_dataset(
    test_raw_path: str | Path = "data/raw/gsm8k_test.jsonl",
    test_output_path: str | Path = "data/processed/pal_test.jsonl",
) -> dict[str, int]:
    """
    构建 PAL 测试集并输出到 JSONL。

    输入：
    - test_raw_path: 原始测试集路径（jsonl）。
    - test_output_path: 输出路径（jsonl）。

    返回：
    - dict[str, int]: raw 与 processed 统计。
    """

    test_raw_path = Path(test_raw_path)
    # 输入文件不存在时直接报错，便于排查配置问题。
    if not test_raw_path.exists():
        raise FileNotFoundError(f"测试原始文件不存在: {test_raw_path}")

    # 全量转换并写出。
    test_records = _read_jsonl(test_raw_path)
    processed_records = [_build_test_record(record) for record in test_records]
    processed_count = _write_jsonl(processed_records, test_output_path)
    return {"raw": len(test_records), "processed": processed_count}