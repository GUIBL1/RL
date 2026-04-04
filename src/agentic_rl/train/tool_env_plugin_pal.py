from __future__ import annotations

# 文件功能：
# 1) 提供 GRPO + GYM 的 Python 工具环境 `python_tool_env`。
# 2) 提供结果奖励函数 `tool_reward`，用于校验 Final Answer 是否正确。
# 3) 通过 ms-swift 插件注册表完成环境与奖励函数注册。

import json
import re
from typing import Any

from swift.rewards import ORM, orms
from swift.rollout.gym_env import Env, envs
from swift.rollout.multi_turn import MultiTurnScheduler, multi_turns

from agentic_rl.tools.python_tool import run_python_tool_from_tool_call_content


FINAL_ANSWER_PATTERN = re.compile(r"Final\s*Answer\s*:\s*(.+)", re.IGNORECASE)
TOOL_CALL_HEADER_PATTERN = re.compile(r"Tool\s*Call\s*:\s*", re.IGNORECASE)
THOUGHT_HEADER_PATTERN = re.compile(r"Thought\s*:\s*", re.IGNORECASE)
USE_TOOL_HEADER_PATTERN = re.compile(r"Use\s*Tool\s*:\s*", re.IGNORECASE)


def _extract_balanced_json_object(text: str, start_index: int) -> str:
    """
    从指定位置开始提取首个平衡的大括号 JSON 对象。

    输入：
    - text: 原始文本。
    - start_index: 开始扫描位置。

    返回：
    - str: 提取到的 JSON 对象字符串；失败返回空串。
    """

    open_index = text.find("{", start_index)
    if open_index < 0:
        return ""

    depth = 0
    in_string = False
    escaped = False

    for index in range(open_index, len(text)):
        char = text[index]

        if escaped:
            escaped = False
            continue

        if char == "\\":
            escaped = True
            continue

        if char == '"':
            in_string = not in_string
            continue

        if in_string:
            continue

        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[open_index:index + 1]

    return ""


def _normalize_answer(value: str) -> str:
    """
    归一化答案文本，减少格式差异带来的误判。

    输入：
    - value: 原始答案文本。

    返回：
    - str: 归一化后的答案。
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


def _extract_final_answer(text: str) -> str:
    """
    从模型输出中抽取 Final Answer 字段。

    输入：
    - text: 模型回复全文。

    返回：
    - str: 抽取到的答案；未匹配返回空串。
    """

    match = FINAL_ANSWER_PATTERN.search(text)
    if not match:
        return ""
    answer = match.group(1).strip()
    # 重要逻辑：只截取当前行，避免把后续段落吞进答案。
    return answer.splitlines()[0].strip()


def _extract_tool_call_content(text: str) -> str:
    """
    从模型输出中抽取 Tool Call JSON 字符串。

    输入：
    - text: 模型回复全文。

    返回：
    - str: Tool Call JSON；未匹配返回空串。
    """

    match = TOOL_CALL_HEADER_PATTERN.search(text)
    if not match:
        return ""

    payload = _extract_balanced_json_object(text, match.end())
    if not payload:
        return ""

    # 兼容 markdown 围栏场景，避免 JSON 解析失败。
    if payload.startswith("```"):
        lines = payload.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        payload = "\n".join(lines).strip()

    return payload


def _extract_final_answer_from_messages(messages: Any) -> str:
    """
    从多轮 messages 中抽取“最后一轮可用终答”。

    输入：
    - messages: 轨迹消息列表。

    返回：
    - str: 最后一轮 Final Answer；未匹配返回空串。
    """

    if not isinstance(messages, list):
        return ""

    for message in reversed(messages):
        if not isinstance(message, dict):
            continue
        if str(message.get("role", "")).strip().lower() != "assistant":
            continue
        content = str(message.get("content", "") or "")
        answer = _extract_final_answer(content)
        if answer:
            return answer
    return ""


def _pick_last_trajectory_sample(request_id: str, trajectory_inputs: Any) -> dict[str, Any] | None:
    """
    从同一 request_id 的轨迹样本中选择“最后一轮”样本。

    选择规则：
    1) 优先 `rollout_infos.num_turns` 更大的样本；
    2) 次优先 messages 更长的样本；
    3) 若仍相同，取出现位置更靠后的样本。
    """

    if not isinstance(trajectory_inputs, dict):
        return None
    candidates = trajectory_inputs.get(request_id)
    if not isinstance(candidates, list) or not candidates:
        return None

    def _rank(item: tuple[int, Any]) -> tuple[int, int, int]:
        index, sample = item
        if not isinstance(sample, dict):
            return (-1, -1, index)
        info = sample.get("rollout_infos", {})
        num_turns = 0
        if isinstance(info, dict):
            try:
                num_turns = int(info.get("num_turns", 0) or 0)
            except Exception:  # noqa: BLE001
                num_turns = 0
        messages = sample.get("messages", [])
        message_count = len(messages) if isinstance(messages, list) else 0
        return (num_turns, message_count, index)

    _, best = max(enumerate(candidates), key=_rank)
    return best if isinstance(best, dict) else None


def _collect_tool_stats_from_trajectory(request_id: str, trajectory_inputs: Any) -> tuple[bool, bool]:
    """
    汇总同轨迹工具状态。

    返回：
    - tuple[bool, bool]: (是否调用过工具, 是否有成功工具调用)
    """

    if not isinstance(trajectory_inputs, dict):
        return False, False
    samples = trajectory_inputs.get(request_id)
    if not isinstance(samples, list):
        return False, False

    called = False
    success = False
    for sample in samples:
        if not isinstance(sample, dict):
            continue
        info = sample.get("rollout_infos", {})
        if not isinstance(info, dict):
            continue
        called = called or bool(info.get("tool_called", False))
        success = success or bool(info.get("tool_exec_success", False))
    return called, success


class PythonToolEnv(Env):
    """
    基于 ms-swift GYM 接口的 Python 工具环境。

    设计目标：
    - 接收模型输出中的 Tool Call。
    - 调用本地 python_tool 执行并把结果作为下一轮 observation。
    - 当模型输出 Final Answer 时结束轨迹。
    """

    def __init__(self, env_config: dict[str, Any]):
        """
        初始化环境状态。

        输入：
        - env_config: 数据集中传入的环境配置。
        """

        super().__init__(env_config)
        self.tool_name = str(env_config.get("tool_name", "python"))
        self.ground_truth = ""
        self.tool_called = False
        self.tool_exec_success = False
        self.last_tool_response = ""

    async def reset(self, config) -> tuple[str, dict[str, Any], str]:
        """
        重置环境并返回首轮观测。

        输入：
        - config: RolloutInferRequest，包含 messages/data_dict。

        返回：
        - observation: 首轮 user 问题文本。
        - info: 调试信息。
        - system_message: 轨迹系统提示词。
        """

        data_dict = getattr(config, "data_dict", {}) or {}
        messages = getattr(config, "messages", []) or []

        self.ground_truth = str(data_dict.get("ground_truth", "")).strip()
        self.tool_called = False
        self.tool_exec_success = False
        self.last_tool_response = ""

        # 重要逻辑：优先复用数据集中的 system/user，保证训练输入与清洗数据一致。
        system_message = ""
        observation = ""
        for message in messages:
            role = str(message.get("role", "")).strip().lower()
            content = str(message.get("content", "")).strip()
            if role == "system" and not system_message:
                system_message = content
            if role == "user" and not observation:
                observation = content

        if not system_message:
            system_message = f'"tools": [{{"type":"function","function":{{"name":"{self.tool_name}"}}}}]'

        return observation, {"env": "python_tool_env"}, system_message

    async def step(self, action) -> tuple[str, float, bool, dict[str, Any]]:
        """
        执行一步环境交互。

        输入：
        - action: 当前消息历史（最后一条为模型回复）。

        返回：
        - next_observation: 下一轮 user 观测。
        - reward: 当前步奖励（本环境置 0，主奖励交给 ORM）。
        - done: 是否终止。
        - info: 轨迹信息，用于日志与奖励函数读取。
        """

        if not action:
            return "", 0.0, True, {"parse_error": "empty_action"}

        model_text = str(action[-1].get("content", "")).strip()
        info: dict[str, Any] = {
            "tool_called": self.tool_called,
            "tool_exec_success": self.tool_exec_success,
        }

        final_answer = _extract_final_answer(model_text)
        if final_answer:
            info.update(
                {
                    "final_answer": final_answer,
                    "tool_called": self.tool_called,
                    "tool_exec_success": self.tool_exec_success,
                    "ground_truth": self.ground_truth,
                }
            )
            return "", 0.0, True, info

        tool_call_content = _extract_tool_call_content(model_text)
        if not tool_call_content:
            info["parse_error"] = "missing_tool_call_or_final_answer"
            return "", 0.0, True, info

        self.tool_called = True
        # 重要逻辑：只负责抽取 Tool Call JSON，具体解析/执行统一交给 python_tool 入口函数。
        tool_response = run_python_tool_from_tool_call_content(tool_call_content)
        self.last_tool_response = tool_response
        try:
            tool_payload = json.loads(tool_response)
            self.tool_exec_success = bool(tool_payload.get("success", False))
            if not self.tool_exec_success:
                info["tool_error"] = str(tool_payload.get("error", "tool_execution_failed"))
        except Exception:  # noqa: BLE001
            self.tool_exec_success = False
            info["tool_error"] = "invalid_tool_response_json"

        info["tool_called"] = self.tool_called
        info["tool_exec_success"] = self.tool_exec_success
        info["tool_response"] = tool_response
        return tool_response, 0.0, False, info

    async def close(self):
        """关闭环境资源。"""

        return None


class MyToolReward(ORM):
    """
        基于“百分比 × 权重”的 PAL 奖励函数（含分级惩罚项）。

        设计目标：
        - 正向奖励严格使用“子分百分比 × 权重”的形式，便于解释与调参。
        - 正确性采用严格匹配，保证终答信号干净。
        - 简洁度采用 GRPO 组内相对分，贴合组内比较机制。
        - 使用分级惩罚项抑制严重错误轨迹。

        评分组成（总分裁剪到 [0, 1]）：
        - 正确性分 correctness in [0, 0.7]
            - 精确匹配：+100% * 0.7
            - 其它：+0% * 0.7
        - 格式分 format in [0, 0.1]
            - Thought: +20% * 0.1
            - Use Tool: +20% * 0.1
            - Tool Call(JSON 可提取): +30% * 0.1
            - Final Answer 格式: +30% * 0.1
        - 工具分 tool in [0, 0.1]
            - 调用工具: +30% * 0.1
            - 执行成功: +70% * 0.1
        - 简洁度分 concise in [0, 0.1]
            - 组内相对简洁度：同组内越短，得分越高（按 completion 全文 token 计数）

        惩罚项（直接扣分）：
        - parse_error: -0.05
        - 工具调用但执行失败: -0.05
        - 没有调用工具: -0.05
        - 绝对过长惩罚：>320 token 开始线性扣分，>=600 token 扣满 -0.10
        - 其他分级惩罚（错误越严重扣越多）：
          - 缺失 Final Answer: -0.08
          - 出现 Tool Call 头但 JSON 不可提取: -0.06
          - 有 Use Tool 但没有有效 Tool Call: -0.04
    """

    # 四项正向奖励权重（总和=1.0）
    CORRECTNESS_WEIGHT = 0.7
    FORMAT_WEIGHT = 0.10
    TOOL_WEIGHT = 0.10
    CONCISE_WEIGHT = 0.10

    # 固定惩罚项（直接扣分）
    PENALTY_PARSE_ERROR = 0.05
    PENALTY_TOOL_FAILED = 0.05
    PENALTY_NO_TOOL_CALLED = 0.05
    PENALTY_MISSING_FINAL_ANSWER = 0.08
    PENALTY_TOOL_CALL_HEADER_WITH_INVALID_JSON = 0.06
    PENALTY_USE_TOOL_WITHOUT_VALID_TOOL_CALL = 0.04

    # 绝对过长惩罚（线性区间）
    OVERLENGTH_START_TOKENS = 320
    OVERLENGTH_MAX_TOKENS = 600
    OVERLENGTH_MAX_PENALTY = 0.10

    @staticmethod
    def _count_completion_tokens(text: str) -> int:
        """按 completion 全文粗粒度计 token，用于组内相对简洁度与超长惩罚。"""

        return len(re.findall(r"[a-zA-Z0-9_]+", str(text or "")))

    def _accuracy_score(self, prediction: str, ground_truth: str) -> float:
        """
        计算答案正确性得分。

        规则：
        - 归一化后字符串完全一致 -> 1.0
        - 其它情况 -> 0.0

        说明：
        - 按当前 PAL 训练集（ground_truth 为确定数值）采用“严格命中”更稳定。
        - 去掉相对误差可避免“近似错答”被过度奖励，提升终答可判别性。
        """

        pred_norm = _normalize_answer(prediction)
        gt_norm = _normalize_answer(ground_truth)
        if pred_norm and pred_norm == gt_norm:
            return 1.0
        return 0.0

    @staticmethod
    def _format_score(completion_text: str, has_final_answer: bool) -> float:
        """
        计算格式分（0~1），再由外部乘 FORMAT_WEIGHT。

        要点：
        - Thought / Use Tool / Tool Call / Final Answer 各自提供细粒度增量。
        - Tool Call 同时要求可提取 JSON，避免仅模板匹配。
        """

        text = str(completion_text or "")
        score = 0.0

        if THOUGHT_HEADER_PATTERN.search(text):
            score += 0.20

        if USE_TOOL_HEADER_PATTERN.search(text):
            score += 0.20

        tool_call_content = _extract_tool_call_content(text)
        if tool_call_content:
            score += 0.30

        if has_final_answer:
            score += 0.30

        return min(1.0, max(0.0, score))

    @staticmethod
    def _tool_score(tool_called: bool, tool_exec_success: bool) -> float:
        """
        计算工具分（0~1），再由外部乘 TOOL_WEIGHT。

        分档：
        - 调用工具: 0.30
        - 工具执行成功: +0.70
        """

        score = 0.0
        if tool_called:
            score += 0.30
        if tool_exec_success:
            score += 0.70
        return min(1.0, max(0.0, score))

    def _build_group_relative_concise_scores(
        self,
        completions: list[str],
        request_ids: list[Any],
    ) -> list[float]:
        """
        计算组内相对简洁度分（0~1）。

        规则：
        - 同组内按 completion 全文 token 数比较，越短得分越高。
        - 公式：score = (max_len - cur_len) / (max_len - min_len)
        - 若组内长度一致，则全组得 1.0。

        分组键：
        - 有 request_id 时按 request_id 分组；
        - 无 request_id 时将全部样本视为同一组。
        """

        group_to_indices: dict[str, list[int]] = {}
        for index, request_id_value in enumerate(request_ids):
            if isinstance(request_id_value, str) and request_id_value:
                group_key = request_id_value
            else:
                group_key = "__global_group__"
            group_to_indices.setdefault(group_key, []).append(index)

        token_lengths = [self._count_completion_tokens(text) for text in completions]
        scores = [1.0 for _ in completions]

        for indices in group_to_indices.values():
            lengths = [token_lengths[idx] for idx in indices]
            min_len = min(lengths)
            max_len = max(lengths)
            if max_len <= min_len:
                for idx in indices:
                    scores[idx] = 1.0
                continue

            denom = float(max_len - min_len)
            for idx in indices:
                scores[idx] = max(0.0, min(1.0, (max_len - token_lengths[idx]) / denom))

        return scores

    def _overlength_penalty(self, completion_text: str) -> float:
        """
        计算绝对过长惩罚。

        规则：
        - <=320 token: 0
        - >=600 token: 扣满 0.10
        - 中间线性插值。
        """

        token_count = self._count_completion_tokens(completion_text)
        if token_count <= self.OVERLENGTH_START_TOKENS:
            return 0.0
        if token_count >= self.OVERLENGTH_MAX_TOKENS:
            return self.OVERLENGTH_MAX_PENALTY

        ratio = (token_count - self.OVERLENGTH_START_TOKENS) / (
            self.OVERLENGTH_MAX_TOKENS - self.OVERLENGTH_START_TOKENS
        )
        return self.OVERLENGTH_MAX_PENALTY * ratio

    def _penalty_value(
        self,
        completion_text: str,
        has_final_answer: bool,
        tool_called: bool,
        tool_exec_success: bool,
        info: dict[str, Any],
    ) -> float:
        """计算总惩罚值（直接扣分）。"""

        penalty = 0.0

        parse_error = str(info.get("parse_error", "")).strip()
        if parse_error:
            penalty += self.PENALTY_PARSE_ERROR

        if tool_called and not tool_exec_success:
            penalty += self.PENALTY_TOOL_FAILED

        if not tool_called:
            penalty += self.PENALTY_NO_TOOL_CALLED

        if not has_final_answer:
            penalty += self.PENALTY_MISSING_FINAL_ANSWER

        tool_header_exists = bool(TOOL_CALL_HEADER_PATTERN.search(str(completion_text or "")))
        tool_call_content = _extract_tool_call_content(str(completion_text or ""))
        use_tool_exists = bool(USE_TOOL_HEADER_PATTERN.search(str(completion_text or "")))

        # 更严重：声明了 Tool Call 但无法抽取 JSON。
        if tool_header_exists and not tool_call_content:
            penalty += self.PENALTY_TOOL_CALL_HEADER_WITH_INVALID_JSON

        # 次严重：声明 Use Tool 但没有有效 Tool Call。
        if use_tool_exists and not tool_call_content:
            penalty += self.PENALTY_USE_TOOL_WITHOUT_VALID_TOOL_CALL

        penalty += self._overlength_penalty(completion_text)
        return max(0.0, penalty)

    @staticmethod
    def _resolve_format_text(
        completion_text: str,
        request_id_value: Any,
        trajectory_inputs: Any,
    ) -> str:
        """
        为格式分解析最合适的文本。

        规则：
        - 默认使用当前 completion 文本。
        - 若存在 trajectory_inputs，则尝试定位当前 request 的样本并拼接 assistant 历史。
        """

        base_text = str(completion_text or "")
        if not isinstance(request_id_value, str) or not request_id_value:
            return base_text
        if not isinstance(trajectory_inputs, dict):
            return base_text

        samples = trajectory_inputs.get(request_id_value)
        if not isinstance(samples, list):
            return base_text

        fallback_text = ""
        for sample in reversed(samples):
            if not isinstance(sample, dict):
                continue
            messages = sample.get("messages", [])
            if not isinstance(messages, list):
                continue

            assistant_contents = []
            for message in messages:
                if not isinstance(message, dict):
                    continue
                if str(message.get("role", "")).strip().lower() != "assistant":
                    continue
                content = str(message.get("content", "") or "").strip()
                if content:
                    assistant_contents.append(content)

            if not assistant_contents:
                continue

            merged = "\n".join(assistant_contents)
            if merged and not fallback_text:
                fallback_text = merged

            # 优先选择包含当前 completion 的样本，避免同 request 的多候选串扰。
            if base_text and base_text in merged:
                return merged

        return fallback_text or base_text

    def __call__(self, completions, ground_truth=None, rollout_infos=None, request_id=None, trajectory_inputs=None,
                 **kwargs):
        """
        计算每条 completion 的奖励。

        输入：
        - completions: 模型最终输出列表。
        - ground_truth: 数据集透传的标准答案列（可能是 list 或 None）。
        - rollout_infos: 多轮环境返回的信息列表。

        返回：
        - list[float]: 每条样本的奖励。
        """

        infos = rollout_infos or [{} for _ in range(len(completions))]
        request_ids = request_id if isinstance(request_id, list) else [None] * len(completions)
        completion_texts = [str(completion) for completion in completions]
        concise_scores = self._build_group_relative_concise_scores(completion_texts, request_ids)
        rewards: list[float] = []

        for index, completion_text in enumerate(completion_texts):
            info = infos[index] if index < len(infos) else {}
            if not isinstance(info, dict):
                info = {}

            request_id_value = request_ids[index] if index < len(request_ids) else None
            format_text = self._resolve_format_text(completion_text, request_id_value, trajectory_inputs)
            answer = _extract_final_answer(completion_text)
            has_final_answer = bool(answer)

            gt_value = None
            if isinstance(ground_truth, list):
                if index < len(ground_truth):
                    gt_value = ground_truth[index]
            elif ground_truth is not None:
                gt_value = ground_truth
            if gt_value is None:
                gt_value = info.get("ground_truth", "")
            gt_text = str(gt_value or "")

            # 优先从 trajectory_inputs 聚合多轮工具状态，避免只看最后一轮导致漏计。
            tool_called = bool(info.get("tool_called", False))
            tool_exec_success = bool(info.get("tool_exec_success", False))
            if isinstance(request_id_value, str) and request_id_value:
                called_from_traj, success_from_traj = _collect_tool_stats_from_trajectory(
                    request_id_value,
                    trajectory_inputs,
                )
                tool_called = tool_called or called_from_traj
                tool_exec_success = tool_exec_success or success_from_traj

            accuracy = self._accuracy_score(answer, gt_text) if has_final_answer else 0.0
            correctness_component = self.CORRECTNESS_WEIGHT * accuracy
            format_component = self.FORMAT_WEIGHT * self._format_score(format_text, has_final_answer)
            tool_component = self.TOOL_WEIGHT * self._tool_score(tool_called, tool_exec_success)
            concise_component = self.CONCISE_WEIGHT * concise_scores[index]

            penalty_component = self._penalty_value(
                completion_text=completion_text,
                has_final_answer=has_final_answer,
                tool_called=tool_called,
                tool_exec_success=tool_exec_success,
                info=info,
            )

            total_score = correctness_component + format_component + tool_component + concise_component - penalty_component
            rewards.append(min(1.0, max(0.0, total_score)))

        return rewards


class PythonToolScheduler(MultiTurnScheduler):
    """
    基于 MultiTurnScheduler 的工具调用多轮调度器。

    设计目标：
    - 在 `swift rlhf` 训练路径中实现多轮工具交互。
    - 每轮若检测到 Tool Call，则执行工具并将结果作为下一轮 user 观测。
    - 若检测到 Final Answer 或无效格式，则终止当前轨迹。
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tool_name = "python"

    def check_finished(self, infer_request, response_choice, current_turn: int) -> bool:
        completion = str(response_choice.message.content or "")
        if _extract_final_answer(completion):
            return True

        tool_call_content = _extract_tool_call_content(completion)
        if tool_call_content:
            return super().check_finished(infer_request, response_choice, current_turn)

        # 输出既不是 Tool Call 也不是 Final Answer，视为当前轨迹终止。
        return True

    def step(self, infer_request, response_choice, current_turn: int) -> dict[str, Any]:
        completion = str(response_choice.message.content or "")
        tool_call_content = _extract_tool_call_content(completion)

        info: dict[str, Any] = {
            "tool_called": False,
            "tool_exec_success": False,
        }

        if not tool_call_content:
            info["parse_error"] = "missing_tool_call"
            return {
                "infer_request": infer_request,
                "response_token_ids": response_choice.token_ids,
                "response_loss_mask": [1] * len(response_choice.token_ids),
                "rollout_infos": info,
            }

        info["tool_called"] = True

        # 重要逻辑：调度器只做 Tool Call 抽取；解析与执行由 python_tool 统一处理。
        tool_response = run_python_tool_from_tool_call_content(tool_call_content)

        try:
            tool_payload = json.loads(tool_response)
            info["tool_exec_success"] = bool(tool_payload.get("success", False))
            if not info["tool_exec_success"]:
                info["tool_error"] = str(tool_payload.get("error", "tool_execution_failed"))
        except Exception:  # noqa: BLE001
            info["tool_exec_success"] = False
            info["tool_error"] = "invalid_tool_response_json"

        info["tool_response"] = tool_response

        # 将工具观测作为下一轮 user 输入，形成多轮交互。
        # MultiTurnScheduler.run 会保留已有 messages 历史并追加 assistant completion，
        # 这里只需补充当前轮 observation（tool_response）。
        infer_request.messages.append({"role": "user", "content": tool_response})

        return {
            "infer_request": infer_request,
            "response_token_ids": response_choice.token_ids,
            "response_loss_mask": [1] * len(response_choice.token_ids),
            "rollout_infos": info,
        }


# 重要逻辑：swift 4.x 通过全局映射注册插件，而不是 register_* 装饰器。
envs["python_tool_env"] = PythonToolEnv
orms["tool_reward"] = MyToolReward
multi_turns["python_tool_scheduler"] = PythonToolScheduler
