import os
import json
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TicketRequest(BaseModel):
    content: str


@app.get("/")
def home():
    return {
        "message": "客服工单 AI 分类助手后端已启动"
    }


@app.post("/classify-ticket")
def classify_ticket(request: TicketRequest):
    content = request.content.strip()

    if content == "":
        raise HTTPException(status_code=400, detail="工单内容不能为空。")

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise HTTPException(status_code=500, detail="缺少 DEEPSEEK_API_KEY，请检查 .env 文件。")

    prompt = f"""
你是一个企业客服工单分类助手。

请根据客户提交的问题，判断这个工单的类型、紧急程度、客户情绪、处理部门、是否需要人工介入，并生成一段客服回复建议。

工单类型只能从下面选择：
订单问题、支付问题、退款问题、发票问题、物流问题、账号问题、产品咨询、技术故障、投诉建议、其他

紧急程度只能从下面选择：
低、中、高

客户情绪只能从下面选择：
平静、着急、不满、愤怒

处理部门只能从下面选择：
客服一线、财务、物流、技术支持、售后、销售、人工主管

请严格返回 JSON，不要返回多余解释。

客户问题：
{content}

返回格式：
{{
  "category": "支付问题",
  "priority": "高",
  "department": "财务",
  "sentiment": "愤怒",
  "need_human": true,
  "reason": "判断原因",
  "suggested_reply": "客服回复建议"
}}
"""

    response = client.chat.completions.create(
        model="deepseek-v4-flash",
        messages=[
            {
                "role": "system",
                "content": "你是一个专业的企业客服工单分析助手，只输出 JSON。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        stream=False
    )

    ai_text = response.choices[0].message.content

    try:
        result = json.loads(ai_text)
    except json.JSONDecodeError:
        raise HTTPException(
            status_code=500,
            detail={
                "message": "AI 返回的不是标准 JSON。",
                "raw_response": ai_text
            }
        )

    now = datetime.now()
    ticket_id = f"TICKET-{now.strftime('%Y%m%d%H%M%S')}"

    return {
        "ticket_id": ticket_id,
        "status": "待处理",
        "created_at": now.strftime("%Y-%m-%d %H:%M:%S"),
        "input": content,
        "result": result
    }