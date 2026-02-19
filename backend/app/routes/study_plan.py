from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from ..schemas import APIErrorResponse, StudyPlanRequest, StudyPlanResponse

router = APIRouter(prefix="/api", tags=["study-plan"])


@router.post(
    "/study-plan",
    response_model=StudyPlanResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
        501: {"model": APIErrorResponse},
    },
)
def create_study_plan(payload: StudyPlanRequest) -> StudyPlanResponse:
    raise HTTPException(
        status_code=status.HTTP_501_NOT_IMPLEMENTED,
        detail="Study plan generation is not implemented yet.",
    )

