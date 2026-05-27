import asyncio
import uuid
import time
from typing import Dict, Any, Callable
import logging

logger = logging.getLogger(__name__)

class JobManager:
    """
    Asynchronous Background Job Orchestrator.
    Handles background processing for heavy ML operations without blocking the API.
    """
    def __init__(self):
        # In-memory queue fallback (since Redis might not be available in HF Spaces)
        self.jobs: Dict[str, Dict[str, Any]] = {}

    def create_job(self) -> str:
        job_id = str(uuid.uuid4())
        self.jobs[job_id] = {
            "status": "queued",
            "progress": 0,
            "result": None,
            "error": None,
            "start_time": time.time(),
            "end_time": None
        }
        return job_id

    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        if job_id not in self.jobs:
            return {"status": "not_found"}
            
        job = self.jobs[job_id]
        runtime = (job.get("end_time") or time.time()) - job["start_time"]
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "progress": job["progress"],
            "runtime_seconds": round(runtime, 2),
            "result": job.get("result"),
            "error": job.get("error")
        }

    async def execute_job(self, job_id: str, task_func: Callable, *args, **kwargs):
        """Runs the task in the background and updates the job dictionary."""
        if job_id not in self.jobs:
            return
            
        self.jobs[job_id]["status"] = "processing"
        self.jobs[job_id]["progress"] = 10
        
        try:
            # If the function is a coroutine, await it, else run in executor
            if asyncio.iscoroutinefunction(task_func):
                result = await task_func(*args, **kwargs)
            else:
                loop = asyncio.get_event_loop()
                # Run CPU-bound sync functions in a threadpool to not block the event loop
                result = await loop.run_in_executor(None, lambda: task_func(*args, **kwargs))
                
            self.jobs[job_id]["status"] = "completed"
            self.jobs[job_id]["progress"] = 100
            self.jobs[job_id]["result"] = result
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}", exc_info=True)
            self.jobs[job_id]["status"] = "failed"
            self.jobs[job_id]["error"] = str(e)
            
        finally:
            self.jobs[job_id]["end_time"] = time.time()

# Global instance for FastAPI injection
job_manager = JobManager()
