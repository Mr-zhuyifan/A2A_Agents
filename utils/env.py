"""
需求：统一加载项目根目录的 .env 文件
思路步骤：
1. 定位项目根目录（本文件的上一级目录）
2. 逐行解析 .env，跳过空行与注释
3. 用 setdefault 写入 os.environ，保证 shell 中已有的环境变量优先
4. 记录加载状态，重复调用不会重复读取

说明：.env 只保留在本地，不提交到仓库；仓库中只保留 .env.example。
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = PROJECT_ROOT / ".env"

_loaded = False


def load_dotenv(path=None) -> bool:
    """把 .env 中的键值对补充进 os.environ（已存在的变量不覆盖）。

    :param path: 可选，指定其他 .env 文件路径；默认使用项目根目录的 .env
    :return: 是否实际读取了文件
    """
    global _loaded
    if _loaded:
        return False
    _loaded = True

    env_path = Path(path) if path else ENV_FILE
    if not env_path.is_file():
        return False

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()

        # 去掉行尾注释（形如 KEY=value  # 说明），但保留引号内的内容
        if " #" in value and not value.startswith(("'", '"')):
            value = value.split(" #", 1)[0].strip()
        # 去掉包裹的引号
        value = value.strip('"').strip("'")

        if key:
            os.environ.setdefault(key, value)

    return True
