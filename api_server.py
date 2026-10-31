"""
需求：SmartVoyage FastAPI后端服务器，提供REST API接口
"""
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import uvicorn

from chat_service import ChatService

app = FastAPI(title="SmartVoyage API", description="基于A2A的旅行智能助手")

# 全局服务实例
chat_service = ChatService()


class ChatRequest(BaseModel):
    message: str


class ProfileRequest(BaseModel):
    profile: dict


@app.get("/")
async def index():
    """返回前端页面"""
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """发送消息，获取回复"""
    response = await chat_service.chat(request.message)
    return {"status": "success", "message": response}


async def sse_generator(message: str):
    """SSE 生成器，逐字流式返回回复"""
    async for chunk in chat_service.chat_stream(message):
        # SSE 格式：每行以 "data: " 开头，用空行分隔
        yield f"data: {json.dumps({'chunk': chunk}, ensure_ascii=False)}\n\n"
    # 发送结束标记
    yield "data: [DONE]\n\n"


@app.post("/api/chat/stream")
async def chat_stream(request: ChatRequest):
    """发送消息，流式获取回复（SSE）"""
    return StreamingResponse(sse_generator(request.message), media_type="text/event-stream")


@app.get("/api/memory")
async def get_memory():
    """获取记忆状态"""
    return {"status": "success", "data": chat_service.get_memory_state()}


@app.post("/api/memory/clear")
async def clear_memory():
    """清空记忆"""
    chat_service.clear_memory()
    return {"status": "success", "message": "记忆已清空"}


@app.post("/api/memory/profile")
async def update_profile(request: ProfileRequest):
    """更新用户偏好"""
    chat_service.update_user_profile(request.profile)
    return {"status": "success", "message": "用户偏好已更新"}


@app.get("/api/agents")
async def get_agents():
    """获取代理卡片信息"""
    return {"status": "success", "data": chat_service.get_agent_cards()}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8085, log_level="info")
