# syntax=docker/dockerfile:1
FROM --platform=linux/amd64 python:3.12-slim

# 设置工作目录
WORKDIR /app

# 设置环境变量
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app:$PYTHONPATH

# 安装系统依赖
#RUN apt-get update && apt-get install -y --no-install-recommends gcc && \
#    rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# 复制项目代码
COPY . .

# 暴露端口
EXPOSE 8085

# 默认命令
CMD ["python", "api_server.py"]
