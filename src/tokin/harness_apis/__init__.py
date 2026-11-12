from .openai_chat import OpenAIChatRequest
from .openai_responses import OpenAIResponsesRequest
from .request import HarnessRequest

HARNESS_APIS: tuple[type[HarnessRequest], ...] = (OpenAIChatRequest, OpenAIResponsesRequest)

__all__ = ["HARNESS_APIS", "HarnessRequest", "OpenAIChatRequest", "OpenAIResponsesRequest"]
