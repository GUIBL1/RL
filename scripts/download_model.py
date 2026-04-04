"""
从 Hugging Face 下载模型的脚本
"""

from huggingface_hub import snapshot_download
import os

# ============ 配置 ============
DEFAULT_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_LOCAL_DIR = "models/Qwen2.5-0.5B-Instruct"
# ==============================


def prompt_with_default(prompt: str, default: str) -> str:
    """读取交互输入，回车时返回默认值。"""

    value = input(f"{prompt}[{default}]: ").strip()
    return value if value else default


MODEL_ID = prompt_with_default("请输入模型ID", DEFAULT_MODEL_ID)
LOCAL_DIR = prompt_with_default("请输入本地保存路径", DEFAULT_LOCAL_DIR)

def download_model():
    """下载模型"""
    if not LOCAL_DIR.strip():
        raise ValueError("本地保存路径不能为空")

    print(f"开始下载模型: {MODEL_ID}")
    print(f"保存位置: {os.path.abspath(LOCAL_DIR)}")
    
    # 创建目录
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    # 下载模型（包含所有相关文件）
    snapshot_download(
        repo_id=MODEL_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False  # Windows用户建议设为False
    )
    
    print(f"\n✅ 下载完成！模型保存在: {os.path.abspath(LOCAL_DIR)}")

if __name__ == "__main__":
    download_model()