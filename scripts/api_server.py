import os
import sys
import uvicorn
import time
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional

# Sync PYTHONPATH
sys.path.append(os.getcwd())

from scripts.rag_app import setup_rag, clean_turkish_text, prepare_e5_input

# 1. INITIALIZE SERVER
app = FastAPI(title="Kanuntek API for Open WebUI")

# Global state
MAIN_VS = None
LLM = None
EMBED_ENGINE = None

@app.on_event("startup")
def startup_event():
    global MAIN_VS, LLM, EMBED_ENGINE
    MAIN_VS, LLM, EMBED_ENGINE = setup_rag()
    print("🚀 Kanuntek API Server is online!")

# 2. OPENAI-COMPATIBLE MODELS
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False

@app.get("/v1/models")
async def list_models():
    return {
        "object": "list",
        "data": [{"id": "kanuntek-legal-rag", "object": "model", "owned_by": "hadi"}]
    }

@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    try:
        # Extract the last user message
        user_query = ""
        for msg in reversed(request.messages):
            if msg.role == "user":
                user_query = msg.content
                break
        
        # DEBUG: Print incoming query length
        print(f"[*] Incoming User Message: {len(user_query)} chars")
        
        # 1. Clean and Prefix Query
        # If the query itself is massive (PDF text), we truncate it for retrieval
        retrieval_q = clean_turkish_text(user_query[:5000]) 
        formatted_q = prepare_e5_input(retrieval_q, is_query=True)
        
        # 2. Retrieve Relevant Context
        print(f"[*] Querying database...")
        docs = MAIN_VS.similarity_search(formatted_q, k=3)
        
        # Aggressively truncate each chunk to ensure we fit in 4k context
        context_parts = []
        for d in docs:
            txt = d.page_content.replace("passage: ", "")
            context_parts.append(txt[:2500]) # Max 2500 chars per doc
            
        context = "\n\n".join(context_parts)
        
        # 3. Format Prompt
        from src.prompts import RAG_PROMPT
        # Strictly limit the total prompt to ~12,000 chars (approx 3500 tokens)
        # This guarantees 500+ tokens for the model response in a 4k window.
        final_prompt = RAG_PROMPT.format(
            chat_history="", 
            context=context, 
            question=user_query[:4000] # Limit user input to 4k chars
        )
        
        if len(final_prompt) > 12000:
            print(f"[*] Force truncating prompt to 12k chars for CPU stability.")
            final_prompt = final_prompt[:12000]

        # 5. Generate Response
        print(f"[*] Generating answer (Prompt: {len(final_prompt)} chars)...")
        start_t = time.time()
        response = LLM.invoke(final_prompt)
        elapsed = time.time() - start_t
        print(f"[*] Answer generated in {elapsed:.2f}s")
        
        return {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": request.model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response},
                "finish_reason": "stop"
            }]
        }
    except Exception as e:
        print(f"CRITICAL ERROR: {str(e)}")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": str(e),
                    "type": "server_error"
                }
            }
        )

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8081)
