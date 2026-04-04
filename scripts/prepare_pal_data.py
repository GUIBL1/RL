from __future__ import annotations

# 文件功能：
# 1) 提供 PAL 的 SFT/GRPO 训练数据与测试数据构建的一站式脚本入口。
# 2) 支持命令行参数与交互式输入两种模式。
# 3) 调用各构建器并输出统计信息。

from pathlib import Path
import argparse
import json
import random
import sys

# 将项目根目录下的 src 添加到 sys.path，避免 ModuleNotFoundError
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from agentic_rl.data.pal_test import build_pal_test_dataset
from agentic_rl.data.grpo_pal_train import build_grpo_pal_train_dataset
from agentic_rl.data.sft_pal_train import build_sft_pal_train_dataset


def prompt_with_default(prompt: str, default: str) -> str:
    """
    显示带默认值的交互提示，并返回用户输入。

    输入：
    - prompt: 提示文本。
    - default: 默认值。

    返回：
    - str: 用户输入；若输入为空则返回默认值。
    """

    # 保持交互输入体验：回车直接使用默认值。
    val = input(f"{prompt} [{default}]: ").strip()
    return val if val != "" else default


def prompt_float_with_default(prompt: str, default: float) -> float:
    """读取浮点型交互输入，空输入回退默认值。"""

    val = input(f"{prompt} [{default}]: ").strip()
    if val == "":
        return float(default)
    return float(val)


def prompt_int_with_default(prompt: str, default: int) -> int:
    """读取整型交互输入，空输入回退默认值。"""

    val = input(f"{prompt} [{default}]: ").strip()
    if val == "":
        return int(default)
    return int(val)


def _read_train_raw_records(path: str | Path) -> list[dict]:
    """
    读取训练原始 JSON 数组。

    输入：
    - path: `gsm8k_train_pal.json` 路径。

    返回：
    - list[dict]: 原始训练样本。
    """

    train_raw_path = Path(path)
    if not train_raw_path.exists():
        raise FileNotFoundError(f"训练原始文件不存在: {train_raw_path}")
    with train_raw_path.open("r", encoding="utf-8") as file:
        records = json.load(file)
    if not isinstance(records, list):
        raise ValueError(f"训练原始文件不是 JSON 数组: {train_raw_path}")
    return records


def _split_non_overlapping_records(records: list[dict], sft_ratio: float, split_seed: int) -> tuple[list[dict], list[dict]]:
    """
    把训练样本按比例切分为 SFT 与 GRPO 两个不相交集合。

    输入：
    - records: 原始训练样本。
    - sft_ratio: SFT 占比，必须在 (0, 1) 内。
    - split_seed: 随机种子，保证可复现。

    返回：
    - tuple[list[dict], list[dict]]: (sft_records, grpo_records)
    """

    if not (0.0 < float(sft_ratio) < 1.0):
        raise ValueError(f"sft_ratio 必须在 (0, 1) 区间，当前为: {sft_ratio}")
    total = len(records)
    if total < 2:
        raise ValueError("训练样本数量不足，至少需要 2 条才能切分为不相交 SFT/GRPO 集")

    indices = list(range(total))
    rng = random.Random(int(split_seed))
    rng.shuffle(indices)

    # 向下取整后再裁剪到 [1, total-1]，确保两侧非空。
    sft_count = int(total * float(sft_ratio))
    sft_count = max(1, min(total - 1, sft_count))

    sft_index_set = set(indices[:sft_count])
    sft_records = [record for i, record in enumerate(records) if i in sft_index_set]
    grpo_records = [record for i, record in enumerate(records) if i not in sft_index_set]
    return sft_records, grpo_records


def parse_args() -> argparse.Namespace:
    """
    支持两种输入方式：
    - 命令行参数（优先）：例如 `--train_raw data/raw/gsm8k_train_pal.json`
    - 交互式终端：若未传入命令行参数且为交互式终端，逐项提示（回车使用默认值）
    非交互式终端且未提供参数时使用默认值（适合 CI）。
    """

    # 定义脚本参数，默认路径直接对应当前项目目录结构。
    parser = argparse.ArgumentParser(description="Process local PAL/GSM8K raw files")
    parser.add_argument("--train_raw", default="data/raw/gsm8k_train_pal.json", help="raw PAL-enhanced train json path")
    parser.add_argument("--test_raw", default="data/raw/gsm8k_test.jsonl", help="raw gsm8k test jsonl path")
    parser.add_argument("--sft_train_out", default="data/processed/sft_pal_train.jsonl", help="processed sft train jsonl path")
    parser.add_argument("--grpo_train_out", default="data/processed/grpo_pal_train.jsonl", help="processed grpo train jsonl path")
    parser.add_argument("--test_out", default="data/processed/pal_test.jsonl", help="processed test jsonl path")
    parser.add_argument("--sft_ratio", type=float, default=0.5, help="SFT 训练集比例，取值 (0,1)")
    parser.add_argument("--split_seed", type=int, default=42, help="SFT/GRPO 划分随机种子")

    args = parser.parse_args()

    # 命令行显式传参时直接使用，优先级最高。
    if len(sys.argv) > 1:
        return args

    # 交互式终端下，逐项询问路径，方便手动运行。
    if sys.stdin.isatty():
        try:
            train_raw = prompt_with_default("PAL增强训练集路径", args.train_raw)
            test_raw = prompt_with_default("测试集路径", args.test_raw)
            sft_train_out = prompt_with_default("SFT训练集输出路径", args.sft_train_out)
            grpo_train_out = prompt_with_default("GRPO训练集输出路径", args.grpo_train_out)
            test_out = prompt_with_default("测试集输出路径", args.test_out)
            sft_ratio = prompt_float_with_default("SFT训练集比例(0~1)", args.sft_ratio)
            split_seed = prompt_int_with_default("SFT/GRPO划分随机种子", args.split_seed)
        except KeyboardInterrupt:
            print("\n已取消。")
            sys.exit(1)
        return argparse.Namespace(
            train_raw=train_raw,
            test_raw=test_raw,
            sft_train_out=sft_train_out,
            grpo_train_out=grpo_train_out,
            test_out=test_out,
            sft_ratio=sft_ratio,
            split_seed=split_seed,
        )

    # 非交互场景（如 CI）且未传参时，回退到默认值。
    print("非交互式终端且未提供参数，使用默认值。")
    return args


def main() -> None:
    """
    主流程：
    1. 解析参数（命令行优先；否则交互式提示；否则默认）
    2. 分别调用训练/测试构建器加工本地 raw 文件
    3. 打印每个 split 的生成数量
    """

    # 先解析参数，再执行训练与测试的转换流程。
    args = parse_args()

    # 读取并切分训练集，确保 SFT/GRPO 使用不相交样本。
    train_records = _read_train_raw_records(args.train_raw)
    sft_records, grpo_records = _split_non_overlapping_records(
        train_records,
        sft_ratio=args.sft_ratio,
        split_seed=args.split_seed,
    )

    # 构建 SFT 训练集（ms-swift agent SFT 格式）。
    sft_train_stats = build_sft_pal_train_dataset(
        train_output_path=args.sft_train_out,
        train_records=sft_records,
    )
    # 构建 GRPO 训练集（messages + ground_truth + env_config）。
    grpo_train_stats = build_grpo_pal_train_dataset(
        train_output_path=args.grpo_train_out,
        train_records=grpo_records,
    )
    # 构建测试集（question-answer 评测格式）。
    test_stats = build_pal_test_dataset(
        test_raw_path=args.test_raw,
        test_output_path=args.test_out,
    )

    print(
        "PAL sft train: "
        f"raw={sft_train_stats['raw']} (ratio={args.sft_ratio}, seed={args.split_seed}), "
        f"processed={sft_train_stats['processed']} -> {Path(args.sft_train_out)}"
    )
    print(
        "PAL grpo train: "
        f"raw={grpo_train_stats['raw']} (ratio={1 - args.sft_ratio}), "
        f"processed={grpo_train_stats['processed']} -> {Path(args.grpo_train_out)}"
    )
    print(
        "PAL test: "
        f"raw={test_stats['raw']} -> {Path(args.test_raw)}, "
        f"processed={test_stats['processed']} -> {Path(args.test_out)}"
    )


if __name__ == "__main__":
    main()