from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from ...services.bible_lookup import (
    InvalidReferenceError,
    PassageNotFoundError,
    PassageProviderError,
)
from ...services.study_plan_service import (
    StudyPlanProviderError,
    StudyPlanService,
    StudyPlanValidationError,
)
from ..schemas import APIErrorResponse, StudyPlanRequest, StudyPlanResponse

router = APIRouter(prefix="/api", tags=["study-plan"])


def get_study_plan_service() -> StudyPlanService:
    return StudyPlanService()


@router.post(
    "/study-plan",
    response_model=StudyPlanResponse,
    responses={
        400: {"model": APIErrorResponse},
        404: {"model": APIErrorResponse},
        502: {"model": APIErrorResponse},
    },
)
def create_study_plan(
    payload: StudyPlanRequest,
    service: StudyPlanService = Depends(get_study_plan_service),
) -> StudyPlanResponse:
    try:
        return service.generate_study_plan(payload)
    except InvalidReferenceError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
    except PassageNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (PassageProviderError, StudyPlanValidationError, StudyPlanProviderError) as exc:
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)) from exc
