from fastapi import APIRouter, Request

from vllm.entrypoints.router.protocol import RouterClassifyRequest

router = APIRouter()


@router.post("/v1/router/classify")
async def router_classify(
    request: RouterClassifyRequest,
    raw_request: Request,
):
    serving = raw_request.app.state.serving_router_classification
    return await serving.classify(request, raw_request)


def attach_router(app):
    app.include_router(router)