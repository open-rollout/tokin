from __future__ import annotations

from abc import ABC, abstractmethod
from http import HTTPStatus
from typing import Any, ClassVar

from pydantic import BaseModel

from ..gateway import ChatRequest, ChatResponse


class HarnessRequest(BaseModel, ABC):
    """A harness's request in one wire API, and the way back to that API's response.

    `path` is where a session serves the API, under its base URL.
    """

    path: ClassVar[str]
    stream: bool = False

    @abstractmethod
    def chat_request(self) -> ChatRequest: ...

    @abstractmethod
    def start_events(self) -> list[str]:
        """What a stream opens with, before any keep-alive: OpenAI's SDK refuses a Responses stream not opened so."""

    @abstractmethod
    def keepalive_event(self) -> str:
        """What a stream sends while its turn generates, so that the harness does not time out on silence."""

    @abstractmethod
    def response(self, reply: ChatResponse) -> BaseModel:
        """The response this API gives for one turn the gateway ran."""

    @abstractmethod
    def response_events(self, reply: ChatResponse) -> list[str]:
        """The response as this API's server-sent events, for a harness that asked to stream."""

    @classmethod
    @abstractmethod
    def error(cls, exc: Exception, status: HTTPStatus) -> dict[str, Any]:
        """The body this API answers a refusal with; a classmethod, as a body failing validation makes no request."""

    @abstractmethod
    def error_events(self, exc: Exception, status: HTTPStatus) -> list[str]:
        """The error as this API's server-sent events, for a stream whose turn fails after its headers are sent."""
