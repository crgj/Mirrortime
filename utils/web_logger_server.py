
import asyncio
import threading
import queue
import json
import time
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import numpy as np
import torch
from pathlib import Path

import psutil

# Global state
app = FastAPI()

class WebLoggerState:
    def __init__(self):
        self.lock = threading.Lock()
        self.total_iterations = 0
        self.current_iteration = 0
        self.training_start_time = 0
        self.max_frames = 300 # Default, updated via set_max_frames
        
        # History data (sparsified)
        self.loss_history = []  # List of {x: iter, y: loss}
        self.psnr_history = []  # List of {x: iter, y: psnr}
        self.point_count_history = [] # List of {x: iter, y: count}
        self.memory_history = [] # List of {x: iter, system: %, gpu: MB}
        
        # Densification History
        self.clone_history = []
        self.split_history = []
        self.pruned_history = []
        
        # Histogram data
        self.lifetime_hist_labels = []
        self.lifetime_hist_counts = []
        
        # Config
        self.config = {}
        
        # Queues for 3D viewer
        self.render_queue = queue.Queue(maxsize=1)
        self.result_queue = queue.Queue(maxsize=1)
        
        # Latest image for direct polling (optional)
        self.latest_image = None

state = WebLoggerState()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/status")
def get_status():
    with state.lock:
        elapsed = time.time() - state.training_start_time if state.training_start_time > 0 else 0
        return {
            "current_iteration": state.current_iteration,
            "total_iterations": state.total_iterations,
            "elapsed_time": elapsed,
            "loss_history": state.loss_history,
            "psnr_history": state.psnr_history,
            "point_count_history": state.point_count_history,
            "memory_history": state.memory_history,
            "clone_history": state.clone_history,
            "split_history": state.split_history,
            "pruned_history": state.pruned_history,
            "lifetime_hist_labels": state.lifetime_hist_labels,
            "lifetime_hist_counts": state.lifetime_hist_counts,
            "config": state.config
        }

@app.websocket("/ws/viewer")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print(f"[Web Logger] Viewer connected: {websocket.client}")
    try:
        while True:
            # Receive camera pose
            data = await websocket.receive_text()
            cam_data = json.loads(data)
            
            # Put in queue for training thread to process
            # We clear the queue first to ensure we process the latest request
            while not state.render_queue.empty():
                try:
                    state.render_queue.get_nowait()
                except queue.Empty:
                    break
            
            state.render_queue.put(cam_data)
            
            # Wait for result (with timeout)
            # Since we can't block async loop with queue.get, we poll or use a future.
            # Simple polling for this script:
            result = None
            for _ in range(200): # Wait up to 2 seconds roughly
                if not state.result_queue.empty():
                    result = state.result_queue.get()
                    break
                await asyncio.sleep(0.01)
            
            if result:
                # Send back binary image
                await websocket.send_bytes(result)
            else:
                # Send error or empty
                await websocket.send_text("render_timeout")
                
    except WebSocketDisconnect:
        print("Viewer disconnected")
    except Exception as e:
        print(f"WS Error: {e}")

# Static files will be mounted later
# app.mount("/", StaticFiles(directory="web_ui", html=True), name="static")

def start_server(port=8080):
    # Mount now to avoid error if dir didn't exist at import time
    if Path("web_ui").exists():
        app.mount("/", StaticFiles(directory="web_ui", html=True), name="static")
    else:
        print("Warning: web_ui directory not found, dashboard will not be served.")

    def run():
        uvicorn.run(app, host="0.0.0.0", port=port, log_level="warning", log_config=None)
    
    t = threading.Thread(target=run, daemon=True)
    t.start()
    print(f"\n{'='*50}")
    print(f"   Note: Web Dashboard Available")
    print(f"   URL: http://localhost:{port}")
    print(f"{'='*50}\n")
    return t

def set_max_frames(max_frames):
    with state.lock:
        state.max_frames = max(1, int(max_frames))

def init_logger(config, total_iterations):
    with state.lock:
        state.config = config
        state.total_iterations = total_iterations
        state.training_start_time = time.time()

def log_metrics(iteration, loss, point_count, psnr=None, lifetime_tensor=None, densification_stats=None):
    with state.lock:
        state.current_iteration = iteration
        
        # Log history every few iterations to save bandwidth
        # Dynamic sampling could be better but fixed interval for now
        should_log = False
        if iteration < 1000:
             should_log = (iteration % 10 == 0)
        else:
             should_log = (iteration % 100 == 0)
             
        if should_log:
            state.loss_history.append({"x": iteration, "y": float(loss)})
            state.point_count_history.append({"x": iteration, "y": int(point_count)})
            if psnr is not None:
                state.psnr_history.append({"x": iteration, "y": float(psnr)})
                
            # Memory Usage
            mem = psutil.virtual_memory()
            gpu_mem = torch.cuda.memory_allocated() / (1024 * 1024) # MB
            state.memory_history.append({
                "x": iteration,
                "system": float(mem.percent),
                "gpu": float(gpu_mem)
            })
            
            if densification_stats:
                state.clone_history.append({"x": iteration, "y": int(densification_stats.get("cloned", 0))})
                state.split_history.append({"x": iteration, "y": int(densification_stats.get("split", 0))})
                state.pruned_history.append({"x": iteration, "y": int(densification_stats.get("pruned", 0))})
            else:
                 # Even if no densification happened this step, we might want to log 0 to keep charts aligned?
                 # Or just sparse log. Let's log 0 if not provided to keep alignment.
                 state.clone_history.append({"x": iteration, "y": 0})
                 state.split_history.append({"x": iteration, "y": 0})
                 state.pruned_history.append({"x": iteration, "y": 0})
            
            # Keep history size manageable (e.g. 50000 points max)
            if len(state.loss_history) > 50000:
                state.loss_history = state.loss_history[::2]
                state.psnr_history = state.psnr_history[::2]
                state.point_count_history = state.point_count_history[::2]
                state.memory_history = state.memory_history[::2]
                state.clone_history = state.clone_history[::2]
                state.split_history = state.split_history[::2]
                state.pruned_history = state.pruned_history[::2]
        
        # Compute histogram occasionally
        if lifetime_tensor is not None and iteration % 200 == 0:
            # lifetime_w is half-width, duration is approx 2 * w (or 4 * w depending on sigma definition)
            # Assuming 'w' is something akin to standard deviation or half-width.
            # We'll plot 2*w.
            # Tensor on GPU
            try:
                # Sample a subset if too large
                if lifetime_tensor.shape[0] > 10000000:
                    indices = torch.randint(0, lifetime_tensor.shape[0], (10000000,), device=lifetime_tensor.device)
                    samples = lifetime_tensor[indices].detach().cpu().numpy().flatten()
                else:
                    samples = lifetime_tensor.detach().cpu().numpy().flatten()
                
                # durations = samples * 2.0 # Approximation <-- user said lifetime distribution, which for 4DGS usually means sum of opacity. 
                # WDD: directly using samples as duration
                durations = samples
                
                # Compute histogram
                # Frame range might be large, clamp to reasonable max (e.g. 1000 frames) or auto
                MAX_FRAME = 30
                hist, bin_edges = np.histogram(durations, bins=30, range=(0, MAX_FRAME))
                
                # Format labels with decimals for precision
                state.lifetime_hist_labels = [f"{bin_edges[i]:.2f}" for i in range(len(hist))]
                state.lifetime_hist_counts = hist.tolist()
            except Exception as e:
                print(f"Histogram error: {e}")

def get_render_request():
    if not state.render_queue.empty():
        try:
            return state.render_queue.get_nowait()
        except queue.Empty:
            return None
    return None

def submit_render_result(image_bytes):
    # Clear old results
    while not state.result_queue.empty():
        try:
            state.result_queue.get_nowait()
        except queue.Empty:
            break
    state.result_queue.put(image_bytes)
