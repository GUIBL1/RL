from __future__ import annotations

# 文件功能：
# 1) 定义 ms-swift Agent 所需的 python 工具协议（tools/tool_call/tool_response）。
# 2) 提供可直接执行 tool_call JSON 字符串的入口函数，返回 tool_response JSON 字符串。
# 3) 提供基础 Python 代码执行能力，供数据构建与推理流程复用。
#
# 调用协议（与下面实现严格一致）：
# - tools 字段（JSON 字符串）可由 `get_python_tool_spec_json()` 获取。
# - tool_call.content 示例：{"name":"python","arguments":{"code":"print(1+2)"}}
# - tool_response.content 示例：{"result":"3","success":true}
# - 若执行失败：{"result":"","success":false,"error":"..."}

import contextlib
import io
import json
import os
import signal
import ast
from dataclasses import dataclass


@dataclass
class PythonExecutionResult:
    """
    描述 Python 代码执行结果。

    字段说明：
    - success: 是否执行成功。
    - stdout: 捕获到的标准输出（已去除首尾空白）。
    - error: 失败时的错误信息。
    """

    success: bool
    stdout: str
    error: str | None = None


PYTHON_TOOL_SPEC: list[dict] = [
    {
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
]


DEFAULT_EXEC_TIMEOUT_SECONDS = float(os.getenv("PYTHON_TOOL_TIMEOUT_SECONDS", "3"))
MAX_CODE_LENGTH = int(os.getenv("PYTHON_TOOL_MAX_CODE_CHARS", "8000"))
MAX_STDOUT_LENGTH = int(os.getenv("PYTHON_TOOL_MAX_STDOUT_CHARS", "4000"))


class _OutputLimitReachedError(RuntimeError):
    """内部异常：stdout 超过限制时用于提前结束执行。"""


class _LimitedStdoutBuffer(io.StringIO):
    """
    限制写入长度的 stdout 缓冲区。

    设计目标：
    - 避免模型生成的工具代码打印超长文本导致内存膨胀。
    - 达到上限后抛出内部异常，快速终止执行。
    """

    def __init__(self, max_chars: int):
        super().__init__()
        self.max_chars = max(1, int(max_chars))
        self.current_chars = 0

    def write(self, text: str) -> int:
        if not text:
            return 0

        remaining = self.max_chars - self.current_chars
        if remaining <= 0:
            raise _OutputLimitReachedError(f"stdout exceeds limit: {self.max_chars} chars")

        chunk = text[:remaining]
        written = super().write(chunk)
        self.current_chars += written

        if len(text) > remaining:
            raise _OutputLimitReachedError(f"stdout exceeds limit: {self.max_chars} chars")

        return written


def _install_timeout(seconds: float):
    """
    安装基于 SIGALRM 的执行超时。

    返回：
    - tuple[bool, Any]: (是否成功启用超时, 旧 signal handler)

    说明：
    - 仅在主线程且平台支持 SIGALRM 时可启用。
    - 若启用失败，调用方应降级为无超时执行。
    """

    if seconds <= 0:
        return False, None

    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        return False, None

    def _timeout_handler(_signum, _frame):
        raise TimeoutError(f"python execution timeout after {seconds:.2f}s")

    try:
        old_handler = signal.getsignal(signal.SIGALRM)
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.setitimer(signal.ITIMER_REAL, float(seconds))
        return True, old_handler
    except Exception:  # noqa: BLE001
        return False, None


def _clear_timeout(enabled: bool, old_handler) -> None:
    """清理超时定时器并恢复旧 signal handler。"""

    if not enabled:
        return
    try:
        signal.setitimer(signal.ITIMER_REAL, 0.0)
        signal.signal(signal.SIGALRM, old_handler)
    except Exception:  # noqa: BLE001
        return


def get_python_tool_spec_json() -> str:
    """
    获取 python 工具规范的 JSON 字符串。

    返回值：
    - 可直接写入 ms-swift 数据集中 `tools` 字段的 JSON 字符串。
    """

    # 使用 ensure_ascii=False 保留中文描述，便于直接查看数据。
    return json.dumps(PYTHON_TOOL_SPEC, ensure_ascii=False)


def parse_python_tool_call_content(content: str) -> str:
    """
    解析并校验 tool_call.content，提取要执行的代码。

    输入：
    - content: `messages` 中 role=`tool_call` 的 content（JSON 字符串）。

    返回值：
    - code 字段内容（字符串）。

    异常：
    - ValueError: 当 JSON 结构、工具名或参数不符合协议时抛出。
    """

    normalized_content = str(content).strip()

    # 兼容 markdown 围栏包裹的 payload。
    if normalized_content.startswith("```"):
        lines = normalized_content.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        while lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        normalized_content = "\n".join(lines).strip()

    # 优先标准 JSON，失败后回退到 python-literal（容忍单引号等轻微偏差）。
    try:
        payload = json.loads(normalized_content)
    except Exception:  # noqa: BLE001
        try:
            payload = ast.literal_eval(normalized_content)
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f"tool_call content 解析失败: {exc}") from exc

    if not isinstance(payload, dict):
        raise ValueError("tool_call content 必须是 JSON object")

    # 校验工具名称，保证协议对齐为 python。
    tool_name = payload.get("name")
    if tool_name != "python":
        raise ValueError(f"不支持的工具名: {tool_name}")

    # 校验 arguments 结构并提取 code。
    arguments = payload.get("arguments")
    if isinstance(arguments, str):
        # 兼容 arguments 被二次序列化成字符串的场景。
        try:
            arguments = json.loads(arguments)
        except Exception:  # noqa: BLE001
            try:
                arguments = ast.literal_eval(arguments)
            except Exception as exc:  # noqa: BLE001
                raise ValueError(f"tool_call.arguments 解析失败: {exc}") from exc

    if not isinstance(arguments, dict):
        raise ValueError("tool_call.arguments 必须是 JSON object")
    code = arguments.get("code")
    if not isinstance(code, str) or not code.strip():
        raise ValueError("tool_call.arguments.code 必须是非空字符串")
    return code


def build_python_tool_response_content(result: PythonExecutionResult) -> str:
    """
    把执行结果封装成 tool_response.content 的 JSON 字符串。

    输入：
    - result: 代码执行结果。

    返回值：
    - 与 ms-swift agent 数据兼容的 JSON 字符串。
    """

    response = {
        "result": result.stdout,
        "success": result.success,
    }
    # 仅在失败时追加 error，避免成功样本出现冗余字段。
    if result.error is not None:
        response["error"] = result.error
    return json.dumps(response, ensure_ascii=False)


def execute_python(code: str) -> PythonExecutionResult:
    """
    执行 Python 代码并捕获标准输出。

    输入：
    - code: 需要执行的 Python 代码文本。

    返回值：
    - PythonExecutionResult: success/stdout/error 三元结果。
    """

    if len(code) > MAX_CODE_LENGTH:
        return PythonExecutionResult(
            success=False,
            stdout="",
            error=f"python code too long: {len(code)} > {MAX_CODE_LENGTH}",
        )

    buffer = _LimitedStdoutBuffer(MAX_STDOUT_LENGTH)
    local_vars: dict[str, object] = {}
    timeout_enabled = False
    old_handler = None
    try:
        timeout_enabled, old_handler = _install_timeout(DEFAULT_EXEC_TIMEOUT_SECONDS)

        # 将 print 输出重定向到内存缓冲，便于返回给模型。
        with contextlib.redirect_stdout(buffer):
            exec(code, {"__builtins__": __builtins__}, local_vars)

        return PythonExecutionResult(success=True, stdout=buffer.getvalue().strip())
    except _OutputLimitReachedError as exc:
        return PythonExecutionResult(success=False, stdout=buffer.getvalue().strip(), error=str(exc))
    except TimeoutError as exc:
        return PythonExecutionResult(success=False, stdout=buffer.getvalue().strip(), error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return PythonExecutionResult(success=False, stdout=buffer.getvalue().strip(), error=str(exc))
    finally:
        _clear_timeout(timeout_enabled, old_handler)


def run_python_tool_from_tool_call_content(tool_call_content: str) -> str:
    """
    统一入口：执行 tool_call.content 并返回 tool_response.content。

    输入：
    - tool_call_content: role=`tool_call` 的 content JSON 字符串。

    返回值：
    - role=`tool_response` 的 content JSON 字符串。

    说明：
    - 该函数用于“模型实际调用工具”的场景，避免仅停留在协议说明。
    """

    try:
        # 先做结构与参数校验，再执行代码。
        code = parse_python_tool_call_content(tool_call_content)
        result = execute_python(code)
    except Exception as exc:  # noqa: BLE001
        # 解析阶段异常也统一转成 tool_response 结构，便于上层消费。
        result = PythonExecutionResult(success=False, stdout="", error=str(exc))
    return build_python_tool_response_content(result)
