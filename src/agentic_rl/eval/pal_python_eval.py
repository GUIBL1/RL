from __future__ import annotations

# 文件功能：
# 1) 读取 PAL 验证集（question/answer）并逐条评测。
# 2) 使用 OpenAI 兼容接口执行“两轮模型调用 + 一次本地 python 工具执行”：
#    - 第一轮：system(工具信息) + user(题目)，抽取并校验 Use Tool + Tool Call。
#    - 工具执行：直接使用 Tool Call JSON 调用 python_tool。
#    - 第二轮：system(工具信息) + user(题目) + assistant(第一轮输出) + user(工具输出)，抽取 Final Answer。
# 3) 对非结构化输出保持鲁棒：未抓到 Tool Call、工具失败、未抓到答案数字，单条判失败并继续。
# 4) 输出逐条结果 JSONL 与汇总指标 JSON，便于分析模型与工具链路质量。

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openai import OpenAI

from agentic_rl.tools.python_tool import (
    parse_python_tool_call_content,
    run_python_tool_from_tool_call_content,
)


PYTHON_TOOL_SPEC_OBJ: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "python",
        "description": "执行 Python 代码并返回标准输出，用于数学计算或数据处理。",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "要执行的 Python 代码，建议包含 print(...) 输出最终结果。",
                }
            },
            "required": ["code"],
        },
    },
}

SYSTEM_PROMPT = json.dumps(PYTHON_TOOL_SPEC_OBJ, ensure_ascii=False)


@dataclass
class EvalSampleResult:
    """
    单条样本评测结果。

    字段说明：
    - question/gold_answer: 原始问题与金标答案。
    - first_model_output/second_model_output: 两轮模型原始文本。
    - tool_call_content: 第一轮提取到的 Tool Call JSON（失败时可能为空）。
    - action_input_code: 从 Tool Call.arguments.code 提取的代码（失败时可能为空）。
    - tool_response_content: python_tool 原始 JSON 返回。
    - pred_answer: 抽取到的最终预测答案（数字文本）。
    - answer_correct: 预测是否与金标等价。
    - used_python_tool/tool_call_success: 是否调用工具、工具是否成功。
    - error: 当前样本失败原因（成功时为 None）。
    """

    question: str
    gold_answer: str
    first_model_output: str
    second_model_output: str
    tool_call_content: str
    action_input_code: str
    tool_response_content: str | None
    pred_answer: str
    answer_correct: bool
    used_python_tool: bool
    tool_call_success: bool
    error: str | None


def prompt_with_default(prompt: str, default: str) -> str:
    """
    显示带默认值的交互输入。

    输入：
    - prompt: 提示文本。
    - default: 默认值。

    返回：
    - str: 用户输入；若直接回车则返回默认值。
    """

    value = input(f"{prompt}[{default}]:").strip()
    return value if value else default


def parse_args() -> argparse.Namespace:
    """
    解析脚本参数。

    规则：
    - 若命令行显式传参（len(sys.argv) > 1），直接使用 argparse 结果。
    - 若无显式参数且处于交互终端，按固定顺序进行交互输入。

    返回：
    - argparse.Namespace: 评测运行配置。
    """

    parser = argparse.ArgumentParser(
        description="PAL eval with OpenAI two-turn + python tool"
    )
    parser.add_argument("--input", default="data/processed/pal_test.jsonl")
    parser.add_argument("--base_url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--api_key", default="EMPTY")
    parser.add_argument("--output_jsonl", default="output/pal_eval_predictions.jsonl")
    parser.add_argument("--summary_json", default="output/pal_eval_summary.json")
    parser.add_argument("--max_samples", type=int, default=100)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--model", default="")
    parser.add_argument("--max_tokens", type=int, default=1024)
    args = parser.parse_args()

    # 自动化脚本/CI 场景：显式参数优先。
    if len(sys.argv) > 1:
        return args

    # 手工运行场景：进入交互输入，避免每次手敲长参数。
    if sys.stdin.isatty():
        try:
            input_path = prompt_with_default("验证集路径", str(args.input))
            base_url = prompt_with_default("模型base_url", str(args.base_url))
            api_key = prompt_with_default("模型api_key", str(args.api_key))
            output_jsonl = prompt_with_default(
                "逐条结果输出路径", str(args.output_jsonl)
            )
            summary_json = prompt_with_default(
                "汇总指标输出路径", str(args.summary_json)
            )
            max_samples = int(
                prompt_with_default("评测样本数限制", str(args.max_samples))
            )
            timeout = int(
                prompt_with_default("HTTP 请求超时时间（秒）", str(args.timeout))
            )
        except KeyboardInterrupt:
            print("\n已取消输入。")
            raise SystemExit(1) from None

        # 交互输入后的配置回填为 Namespace，保持下游统一处理。
        return argparse.Namespace(
            input=input_path,
            base_url=base_url,
            api_key=api_key,
            output_jsonl=output_jsonl,
            summary_json=summary_json,
            max_samples=max_samples,
            timeout=timeout,
            model=args.model,
            max_tokens=args.max_tokens,
        )

    return args


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """
    读取 JSONL 文件并解析为字典列表。

    输入：
    - path: JSONL 文件路径。

    返回：
    - list[dict[str, Any]]: 每行对应一条字典记录。
    """

    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line in file:
            text = line.strip()
            # 跳过空行，避免 JSON 解析异常。
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    """
    写出 JSONL 文件。

    输入：
    - path: 输出路径。
    - rows: 待写出的字典列表。
    """

    output_path = Path(path)
    # 首次运行时自动创建目录，避免路径不存在导致失败。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _extract_tool_call_block(model_output: str) -> str:
    """
    从第一轮模型输出中提取 Tool Call 后的 JSON 字符串。

    输入：
    - model_output: 第一轮模型原始文本。

    返回：
    - str: 提取到的 Tool Call JSON；提取失败返回空串。

    处理策略：
    - 按 "Tool Call:" 定位 JSON 段，并在下一个结构化标记前截断。
    - 兼容 ```json 围栏输出。
    - 若检测到明显污染 token（如 Final Answer/Thought），直接判失败。
    """

    marker = "Tool Call:"
    if marker in model_output:
        chunk = model_output.split(marker, maxsplit=1)[1]
        # 遇到下一段结构标记立即截断，降低误提取风险。
        stop_markers = [
            "\nFinal Answer:",
            "\nThought:",
            "\nUse Tool:",
            "\nObservation:",
            "\nTool Response:",
        ]
        end = len(chunk)
        for stop in stop_markers:
            idx = chunk.find(stop)
            if idx != -1:
                end = min(end, idx)
        tool_call = chunk[:end].strip()
    else:
        return ""

    # 兼容 Markdown 代码块输出。
    if tool_call.startswith("```"):
        lines = tool_call.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        tool_call = "\n".join(lines).strip()

    if not tool_call:
        return ""

    # 明显污染文本直接判失败。
    bad_tokens = ["Final Answer:", "Thought:", "Use Tool:", "Tool Response:"]
    if any(token in tool_call for token in bad_tokens):
        return ""

    return tool_call.strip()


def extract_use_tool(model_output: str) -> str:
    """
    从第一轮模型输出中提取 Use Tool 的工具名。

    输入：
    - model_output: 第一轮模型输出。

    返回：
    - str: 工具名；提取失败返回空串。
    """

    match = re.search(r"Use Tool\s*:\s*([^\n]+)", model_output)
    if not match:
        return ""
    return match.group(1).strip()


def extract_tool_call_and_code(model_output: str) -> tuple[str, str, str]:
    """
    从第一轮模型输出中提取并校验工具调用，返回 Tool Call 与 code。

    输入：
    - model_output: 第一轮模型输出。

    返回：
    - tuple[str, str, str]: (use_tool_name, tool_call_content, action_input_code)。

    异常：
    - ValueError: 当 Use Tool 缺失、Tool Call 缺失、二者不匹配或 Tool Call 非法时抛出。
    """

    use_tool = extract_use_tool(model_output)
    if not use_tool:
        raise ValueError("未捕捉到 Use Tool")

    tool_call_content = _extract_tool_call_block(model_output)
    if not tool_call_content:
        raise ValueError("未捕捉到 Tool Call")

    # 重要逻辑：Use Tool 与 Tool Call.name 必须一致，且当前评测仅允许 python。
    normalized_use_tool = use_tool.strip().lower()
    if normalized_use_tool != "python":
        raise ValueError(f"Use Tool 非 python: {use_tool}")

    try:
        tool_call_payload = json.loads(tool_call_content)
    except Exception as exc:  # noqa: BLE001
        raise ValueError(f"Tool Call 不是合法 JSON: {exc}") from exc

    tool_name = str(tool_call_payload.get("name", "")).strip().lower()
    if tool_name != normalized_use_tool:
        raise ValueError("Use Tool 与 Tool Call.name 不匹配")

    action_input_code = parse_python_tool_call_content(tool_call_content)
    return normalized_use_tool, tool_call_content, action_input_code


def extract_final_answer_number(model_output: str) -> str:
    """
    从第二轮模型输出中提取 Final Answer 后的数字。

    输入：
    - model_output: 第二轮模型原始文本。

    返回：
    - str: 提取到的数字文本；提取失败返回空串。
    """

    # 仅在显式出现 Final Answer: 时提取，避免误读其它数字。
    matches = re.findall(r"Final Answer\s*:\s*(.+)", model_output)
    if not matches:
        return ""

    candidate = matches[-1].strip()
    number_match = re.search(r"[-+]?\d+(?:\.\d+)?", candidate)
    if not number_match:
        return ""
    return number_match.group(0)


def normalize_answer(value: str) -> str:
    """
    归一化答案文本，用于稳健比较。

    输入：
    - value: 原始答案文本。

    返回：
    - str: 归一化结果。

    规则：
    - 去掉逗号、美元符号、末尾句点。
    - 可解析为数字时统一格式（10.0 -> 10）。
    - 非数字统一小写。
    """

    cleaned = str(value).strip().replace(",", "").replace("$", "")
    if cleaned.endswith("."):
        cleaned = cleaned[:-1].strip()
    if re.fullmatch(r"[-+]?\d+(\.\d+)?", cleaned):
        number = float(cleaned)
        if abs(number - round(number)) < 1e-9:
            return str(int(round(number)))
        return f"{number:.12g}"
    return cleaned.lower()


def answers_equal(pred: str, gold: str) -> bool:
    """
    判断预测答案与金标是否等价。

    输入：
    - pred: 预测答案文本。
    - gold: 金标答案文本。

    返回：
    - bool: 等价返回 True，否则 False。
    """

    return normalize_answer(pred) == normalize_answer(gold)


def pick_model_id(client: OpenAI, model_arg: str) -> str:
    """
    选择模型 ID。

    输入：
    - client: OpenAI 客户端。
    - model_arg: 用户显式传入的模型名。

    返回：
    - str: 最终用于推理的模型名。

    规则：
    - 传了 --model 则直接使用。
    - 否则取 models.list() 的第一项。
    """

    if model_arg.strip():
        return model_arg.strip()

    models = [item.id for item in client.models.list().data]
    if not models:
        raise RuntimeError("models.list() 返回为空，无法自动选择模型")
    return models[0]


def call_chat(
    client: OpenAI,
    model: str,
    messages: list[dict[str, str]],
    timeout: int,
    max_tokens: int,
) -> str:
    """
    调用 OpenAI 兼容 chat.completions 并返回文本内容。

    输入：
    - client/model/messages: 接口核心参数。
    - timeout/max_tokens: 推理参数。

    返回：
    - str: 模型输出文本；为空时返回空串。
    """

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        timeout=timeout,
        max_tokens=max_tokens,
        # 评测场景固定温度为 0，尽量减少随机性。
        temperature=0,
    )
    content = resp.choices[0].message.content
    return str(content or "")


def evaluate_samples(
    samples: list[dict[str, Any]],
    client: OpenAI,
    model: str,
    timeout: int,
    max_tokens: int,
) -> list[EvalSampleResult]:
    """
    逐样本执行两轮评测流程。

    输入：
    - samples: 验证集样本，至少包含 question/answer。
    - client/model/timeout/max_tokens: 推理参数。

    返回：
    - list[EvalSampleResult]: 全部样本的评测明细。

    鲁棒性策略：
    - 任意步骤异常仅影响当前样本，不中断整体评测。
    - 抓不到 Tool Call、工具失败、抓不到 Final Answer 数字，均判该条失败。
    """

    results: list[EvalSampleResult] = []
    for sample in samples:
        # 输入样本缺字段时兜底为空串，保证流程不中断。
        question = str(sample.get("question", "")).strip()
        gold_answer = str(sample.get("answer", "")).strip()

        first_model_output = ""
        second_model_output = ""
        tool_call_content = ""
        action_input_code = ""
        tool_response_content: str | None = None
        pred_answer = ""
        used_tool_name = ""
        used_python_tool = False
        tool_call_success = False
        error: str | None = None

        try:
            # 第一轮：system(工具信息) + user(问题)。
            messages_round_1 = [
                {"role": "system", "content": "\"tools\": [" + SYSTEM_PROMPT + "]"},
                {"role": "user", "content": question},
            ]
            first_model_output = call_chat(
                client=client,
                model=model,
                messages=messages_round_1,
                timeout=timeout,
                max_tokens=max_tokens,
            )

            # 抽取并校验 Use Tool + Tool Call，缺失或不匹配则直接判失败。
            used_tool_name, tool_call_content, action_input_code = extract_tool_call_and_code(
                first_model_output
            )

            # 重要逻辑：新格式下 Tool Call 已是完整协议，直接执行，不能再二次封装。
            used_python_tool = used_tool_name == "python"
            tool_response_content = run_python_tool_from_tool_call_content(
                tool_call_content
            )

            try:
                # 工具返回统一要求为 JSON，包含 success/result。
                tool_payload = json.loads(tool_response_content)
                tool_call_success = bool(tool_payload.get("success", False))
                tool_result = str(tool_payload.get("result", "")).strip()
            except Exception as exc:
                raise ValueError(f"工具返回非 JSON 或结构异常: {exc}") from exc

            # 按需求：工具失败或没有结果，当前样本直接判失败。
            if not tool_call_success or not tool_result:
                raise ValueError("python 工具执行失败或未返回结果")

            # 第二轮：system(工具信息) + user(问题) + user(工具输出)。
            messages_round_2 = [
                {"role": "system", "content": "\"tools\": [" + SYSTEM_PROMPT + "]"},
                {"role": "user", "content": question},
                {"role": "assistant", "content": first_model_output},
                {"role": "user", "content": tool_response_content},
            ]
            second_model_output = call_chat(
                client=client,
                model=model,
                messages=messages_round_2,
                timeout=timeout,
                max_tokens=max_tokens,
            )

            # 仅接受 Final Answer 后数字作为最终答案。
            pred_answer = extract_final_answer_number(second_model_output)
            if not pred_answer:
                raise ValueError("未捕捉到 Final Answer 后的数字")

        except Exception as exc:  # noqa: BLE001
            # 单条异常兜底：记录错误并继续后续样本。
            error = str(exc)
            pred_answer = ""

        # 无论成功失败都产出结构化结果，便于离线排障。
        answer_correct = answers_equal(pred_answer, gold_answer)
        results.append(
            EvalSampleResult(
                question=question,
                gold_answer=gold_answer,
                first_model_output=first_model_output,
                second_model_output=second_model_output,
                tool_call_content=tool_call_content,
                action_input_code=action_input_code,
                tool_response_content=tool_response_content,
                pred_answer=pred_answer,
                answer_correct=answer_correct,
                used_python_tool=used_python_tool,
                tool_call_success=tool_call_success,
                error=error,
            )
        )

    return results


def summarize(results: list[EvalSampleResult]) -> dict[str, Any]:
    """
    汇总评测指标。

    输入：
    - results: 逐条评测结果。

    返回：
    - dict[str, Any]: 总量、正确率、工具调用成功率、错误率等统计。
    """

    total = len(results)
    answer_correct = sum(1 for item in results if item.answer_correct)
    used_tool = sum(1 for item in results if item.used_python_tool)
    tool_success = sum(1 for item in results if item.tool_call_success)
    error_count = sum(1 for item in results if item.error)
    return {
        "total": total,
        "answer_correct": answer_correct,
        "answer_accuracy": (answer_correct / total) if total else 0.0,
        "used_python_tool": used_tool,
        "tool_usage_rate": (used_tool / total) if total else 0.0,
        "tool_call_success": tool_success,
        "tool_success_rate": (tool_success / total) if total else 0.0,
        "error_count": error_count,
        "error_rate": (error_count / total) if total else 0.0,
    }


def main() -> None:
    """
    主流程入口。

    流程：
    1) 读取参数与验证集；
    2) 初始化 OpenAI 客户端并确定模型名；
    3) 执行逐样本评测并输出明细与汇总文件。
    """

    args = parse_args()
    rows = read_jsonl(args.input)

    # 支持快速冒烟：max_samples > 0 时只跑前 N 条。
    if int(args.max_samples) > 0:
        rows = rows[: int(args.max_samples)]

    client = OpenAI(api_key=args.api_key, base_url=args.base_url)
    model = pick_model_id(client=client, model_arg=str(args.model))
    print(f"model: {model}")

    results = evaluate_samples(
        samples=rows,
        client=client,
        model=model,
        timeout=int(args.timeout),
        max_tokens=int(args.max_tokens),
    )

    jsonl_rows = [
        {
            "question": item.question,
            "gold_answer": item.gold_answer,
            "pred_answer": item.pred_answer,
            "answer_correct": item.answer_correct,
            "used_python_tool": item.used_python_tool,
            "tool_call_success": item.tool_call_success,
            "tool_call_content": item.tool_call_content,
            "action_input_code": item.action_input_code,
            "tool_response_content": item.tool_response_content,
            "first_model_output": item.first_model_output,
            "second_model_output": item.second_model_output,
            "error": item.error,
        }
        for item in results
    ]
    summary = summarize(results)

    # 分别输出逐条明细与汇总指标。
    write_jsonl(args.output_jsonl, jsonl_rows)
    Path(args.summary_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary_json).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved: {args.output_jsonl}")
    print(f"saved: {args.summary_json}")


if __name__ == "__main__":
    main()
