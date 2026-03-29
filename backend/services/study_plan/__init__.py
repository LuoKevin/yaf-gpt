from .bible_lookup import (
    BibleAPIProvider,
    BibleLookupError,
    BibleProvider,
    InvalidReferenceError,
    PassageData,
    PassageNotFoundError,
    PassageProviderError,
    PassageTooLongError,
    PassageVerse,
)
from .docx_structure import (
    LukeStructureContext,
    LukeStructureExample,
    LukeStructureRetriever,
    parse_luke_reference,
    parse_luke_reference_from_filename,
    parse_luke_structure_doc,
)
from .style_guide import LukeStyleGuide, load_luke_style_guide

__all__ = [
    "build_repair_messages",
    "build_study_plan_messages",
    "BibleAPIProvider",
    "BibleLookupError",
    "BibleProvider",
    "InvalidReferenceError",
    "load_luke_style_guide",
    "LukeStructureContext",
    "LukeStructureExample",
    "LukeStructureRetriever",
    "LukeStyleGuide",
    "parse_luke_reference",
    "parse_luke_reference_from_filename",
    "parse_luke_structure_doc",
    "PassageData",
    "PassageImageProviderError",
    "PassageImageService",
    "PassageNotFoundError",
    "PassageProviderError",
    "PassageTooLongError",
    "PassageVerse",
    "StudyPlanProviderError",
    "StudyPlanService",
    "StudyPlanValidationError",
]


def __getattr__(name: str):
    if name in {"build_repair_messages", "build_study_plan_messages"}:
        from .prompts import build_repair_messages, build_study_plan_messages

        exports = {
            "build_repair_messages": build_repair_messages,
            "build_study_plan_messages": build_study_plan_messages,
        }
        return exports[name]

    if name in {"StudyPlanProviderError", "StudyPlanService", "StudyPlanValidationError"}:
        from .service import StudyPlanProviderError, StudyPlanService, StudyPlanValidationError

        exports = {
            "StudyPlanProviderError": StudyPlanProviderError,
            "StudyPlanService": StudyPlanService,
            "StudyPlanValidationError": StudyPlanValidationError,
        }
        return exports[name]

    if name in {"PassageImageProviderError", "PassageImageService"}:
        from .passage_image_service import PassageImageProviderError, PassageImageService

        exports = {
            "PassageImageProviderError": PassageImageProviderError,
            "PassageImageService": PassageImageService,
        }
        return exports[name]

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
