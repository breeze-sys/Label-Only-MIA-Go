# ==========================================
# Label-Only-MIA 后端服务接口 (最终部署版)
# 对应任务：双机部署支持、API对齐、Batch加速
# ==========================================

import torch
import torch.nn as nn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from contextlib import asynccontextmanager 
import uvicorn
import os
import sys

# 导入原项目的模型定义
from classifier import CNN

# ================= 配置参数 =================
MODEL_ARCH = 'CNN7'       
DATASET_NAME = 'CIFAR10' 

# 【核心修改】：优先从环境变量读取路径，如果没有则使用默认值
# 默认指向目标模型 (Target)，但启动影子服务时可以通过环境变量覆盖
DEFAULT_PATH = 'python_server/CIFAR10/shadow_json_aligned/best_checkpoint_ep.pth'
CHECKPOINT_PATH = os.getenv("MODEL_PATH", DEFAULT_PATH)
MODEL_DROPOUT_ENV = os.getenv("MODEL_DROPOUT")
MODEL_DROPOUT_P = float(os.getenv("MODEL_DROPOUT_P", "0.5"))

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# 图像规格常量
FLATTENED_SIZE = 3072 # 3 * 32 * 32
# ===========================================

model = None


def resolve_dropout(model_path: str) -> bool:
    if MODEL_DROPOUT_ENV is not None:
        return MODEL_DROPOUT_ENV.lower() in {"1", "true", "yes", "on"}
    return "shadow_json_aligned" in model_path

# --- 生命周期管理 ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    print(f"\n======== 服务启动 (Process ID: {os.getpid()}) ========")
    print(f"[*] 检测设备: {DEVICE}")
    print(f"[*] 加载模型路径: {CHECKPOINT_PATH}")
    
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[!] 严重错误：找不到模型文件 {CHECKPOINT_PATH}")
        # 在多进程模式下，这里退出可以防止 worker 挂起
        sys.exit(1)
    
    # 1. 初始化
    model_instance = CNN(
        MODEL_ARCH,
        DATASET_NAME,
        dropout=resolve_dropout(CHECKPOINT_PATH),
        dropout_p=MODEL_DROPOUT_P,
    )
    
    # 2. 加载权重
    try:
        # map_location 确保 CPU/GPU 兼容
        checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        
        # 智能拆包
        state_dict_to_load = None
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict_to_load = checkpoint['state_dict']
        else:
            state_dict_to_load = checkpoint

        # 清洗 module. 前缀
        clean_state_dict = {}
        for k, v in state_dict_to_load.items():
            if k.startswith('module.'):
                clean_state_dict[k[7:]] = v 
            else:
                clean_state_dict[k] = v
        
        model_instance.load_state_dict(clean_state_dict)
        
        # 送入显卡并 Eval 模式 (关键：关闭 Dropout 随机性)
        model_instance.to(DEVICE)
        model_instance.eval() 
        model = model_instance
        print("[+] 权重加载成功！Success！")
        
    except Exception as e:
        print(f"\n[!!!] 模型加载崩溃: {str(e)}")
        sys.exit(1)

    print(f"[+] 接口列表已对齐:")
    print(f"    - POST /predict       (返回 Label + Logits)")
    print(f"    - POST /predict_batch (高并发加速)")
    print("==============================================\n")
    
    yield
    
    print(f"[*] 服务关闭，清理显存...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

app = FastAPI(title="MIA Attack Oracle", lifespan=lifespan)

# --- 协议定义 (对齐 Go 的 types.go) ---
class PredictRequest(BaseModel):
    image: List[float] # [3072]

class PredictResponse(BaseModel):
    label: int           
    logits: List[float] # 原始分数

class BatchPredictRequest(BaseModel):
    images: List[List[float]] 

class BatchPredictResponse(BaseModel):
    labels: List[int]            
    logits_batch: List[List[float]] 

# --- API 实现 ---

# 接口 1: 单图预测 (兼容 /predict 和 /predict_logits)
@app.post("/predict", response_model=PredictResponse)
@app.post("/predict_logits", response_model=PredictResponse) # 别名路由，防备队友写错路径
async def predict(req: PredictRequest):
    if model is None: raise HTTPException(500, "Model not ready")
    if len(req.image) != FLATTENED_SIZE: raise HTTPException(400, "Shape Error")

    try:
        input_tensor = torch.tensor(req.image).float().to(DEVICE).view(1, 3, 32, 32)
        
        with torch.no_grad(): # 必加：防止 OOM
            output = model(input_tensor)
            pred_label = output.argmax(dim=1).item()
            logits_list = output.cpu().squeeze().tolist()
            
            return PredictResponse(label=pred_label, logits=logits_list)
            
    except Exception as e:
        raise HTTPException(500, str(e))

# 接口 2: 批量预测 (支撑 5x11 并发)
@app.post("/predict_batch", response_model=BatchPredictResponse)
async def predict_batch(req: BatchPredictRequest):
    if model is None: raise HTTPException(500, "Model not ready")
    
    batch_size = len(req.images)
    if batch_size == 0: return BatchPredictResponse(labels=[], logits_batch=[])

    try:
        # List -> Tensor -> GPU
        # 这一步对于大 Batch 可能会有点慢，但比起 HTTP RTT 已经很快了
        input_tensor = torch.tensor(req.images).float().to(DEVICE)
        input_tensor = input_tensor.view(-1, 3, 32, 32)
        
        with torch.no_grad(): # 必加：显存保护
            output = model(input_tensor)
            pred_labels = output.argmax(dim=1).cpu().tolist()
            pred_logits = output.cpu().tolist()
            
            return BatchPredictResponse(labels=pred_labels, logits_batch=pred_logits)
            
    except Exception as e:
        raise HTTPException(500, str(e))

# 本地调试用
if __name__ == "__main__":
    # 关键修改：从环境变量读取 PORT，如果没设置则默认用 8000
    import os
    port_str = os.getenv("PORT", "8000") 
    port = int(port_str)
    
    print(f"🚀 服务即将启动在端口: {port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=False)
