"""
SMB AdTech Platform — FastAPI 入口
注册所有路由、中间件、生命周期钩子
"""
import time
import uuid
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from api.routers import campaigns, bidding

# ─────────────────────────────────────────────
# 结构化日志配置
# ─────────────────────────────────────────────
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer(),
    ]
)
logger = structlog.get_logger(__name__)

# ─────────────────────────────────────────────
# Prometheus 指标
# ─────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "http_requests_total", "Total HTTP requests", ["method", "path", "status"]
)
REQUEST_LATENCY = Histogram(
    "http_request_duration_seconds", "HTTP request latency", ["method", "path"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
BID_LATENCY = Histogram(
    "bid_decision_duration_ms", "Bidding decision latency (ms)",
    buckets=[5, 10, 20, 50, 100, 200, 500],
)


# ─────────────────────────────────────────────
# 生命周期：启动 / 关闭钩子
# ─────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用启动和关闭的资源管理"""
    logger.info("app.startup", version=app.version)

    # TODO: 初始化数据库连接池
    # async with database.connect():
    #     yield

    # TODO: 初始化 Redis 连接
    # app.state.redis = await aioredis.create_redis_pool(settings.REDIS_URL)

    # TODO: 预热 ML 模型
    # from ml.models.bidding_model import BiddingModel
    # app.state.bid_model = BiddingModel.load("ml/artifacts/bidding_model.pkl")

    logger.info("app.ready")
    yield

    # 关闭时清理资源
    logger.info("app.shutdown")
    # await app.state.redis.close()


# ─────────────────────────────────────────────
# FastAPI 应用实例
# ─────────────────────────────────────────────
app = FastAPI(
    title="SMB AdTech Platform API",
    description="""
    中小企业广告技术平台 API

    ## 功能模块
    - **Campaigns**: 广告活动 CRUD 管理
    - **Bidding**: 实时竞价（RTB）引擎，目标 P99 < 100ms
    - **Measurement**: 多触点归因分析
    - **ML**: 自动化竞价优化

    ## 认证
    使用 Bearer Token（JWT）认证。
    """,
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ─────────────────────────────────────────────
# 中间件
# ─────────────────────────────────────────────

# CORS（开发环境放开，生产按域名白名单）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "https://app.smb-adtech.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Gzip 压缩（响应体 > 1KB 自动压缩）
app.add_middleware(GZipMiddleware, minimum_size=1024)


@app.middleware("http")
async def request_context_middleware(request: Request, call_next):
    """
    请求上下文中间件：
    - 注入 request_id（链路追踪）
    - 记录请求日志
    - 统计 Prometheus 指标
    """
    request_id = str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(request_id=request_id)

    t0 = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - t0

    # 添加追踪 header
    response.headers["X-Request-ID"] = request_id
    response.headers["X-Process-Time"] = f"{duration:.4f}"

    # Prometheus 指标
    path = request.url.path
    REQUEST_COUNT.labels(request.method, path, response.status_code).inc()
    REQUEST_LATENCY.labels(request.method, path).observe(duration)

    logger.info(
        "http.request",
        method=request.method,
        path=path,
        status=response.status_code,
        duration_ms=round(duration * 1000, 2),
    )
    return response


# ─────────────────────────────────────────────
# 全局异常处理
# ─────────────────────────────────────────────

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error("unhandled_exception", error=str(exc), path=request.url.path, exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error", "request_id": request.headers.get("X-Request-ID")},
    )


# ─────────────────────────────────────────────
# 路由注册
# ─────────────────────────────────────────────

app.include_router(campaigns.router, prefix="/api/v1")
app.include_router(bidding.router, prefix="/api/v1")

# 未来扩展路由（预留）
# app.include_router(analytics.router, prefix="/api/v1")
# app.include_router(measurement.router, prefix="/api/v1")
# app.include_router(auth.router, prefix="/api/v1")


# ─────────────────────────────────────────────
# 内置端点
# ─────────────────────────────────────────────

@app.get("/health", tags=["system"], summary="健康检查")
async def health_check():
    """Kubernetes liveness probe"""
    return {"status": "ok", "version": app.version}


@app.get("/ready", tags=["system"], summary="就绪检查")
async def readiness_check():
    """Kubernetes readiness probe — 检查依赖服务是否就绪"""
    checks = {
        "api": "ok",
        # TODO: "database": await check_db(),
        # TODO: "redis": await check_redis(),
    }
    all_ok = all(v == "ok" for v in checks.values())
    return JSONResponse(
        status_code=200 if all_ok else 503,
        content={"status": "ready" if all_ok else "degraded", "checks": checks},
    )


@app.get("/metrics", tags=["system"], summary="Prometheus 指标")
async def metrics():
    """暴露 Prometheus 格式指标"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# ─────────────────────────────────────────────
# 开发调试入口
# ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
