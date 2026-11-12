from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from http import HTTPStatus
from typing import cast

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, ValidationError
from pydantic_core import InitErrorDetails

from .chat_templates import ChatTemplateError
from .gateway import ChatResponse, Gateway, GatewayError
from .harness_apis import HARNESS_APIS, HarnessRequest, OpenAIChatRequest
from .inference_engines import InferenceEngineError
from .rollout import GenerationParams
from .session import Session

logger = logging.getLogger(__name__)


class UnknownSessionError(GatewayError):
    status_code = HTTPStatus.NOT_FOUND


class SessionRequest(BaseModel, extra="forbid"):
    """The body of `POST /sessions`: what the trainer fixes for every turn of the session; see `Session`."""

    params: GenerationParams = Field(default_factory=lambda: GenerationParams())
    resume_aborted: bool = False


class Server:
    """The gateway over HTTP: `POST /sessions` opens a session, serving OpenAI's chat and responses at its base URL.

    `app` is the ASGI application. Startup waits up to `ready_timeout` seconds for the engine to generate; a stream
    still generating sends its API's keep-alive every `keepalive` seconds.
    """

    def __init__(self, gateway: Gateway, *, ready_timeout: float = 1800, keepalive: float = 30) -> None:
        self.gateway = gateway
        self.ready_timeout = ready_timeout
        self.keepalive = keepalive
        self.sessions: dict[str, Session] = {}
        self.app = FastAPI(lifespan=self.lifespan)
        self.app.add_api_route("/health", self.health, methods=["GET"])
        self.app.add_api_route("/sessions", self.create_session, methods=["POST"], status_code=HTTPStatus.CREATED)
        self.apis = {f"/sessions/{{sid}}/v1{api.path}": api for api in HARNESS_APIS}
        for path, api in self.apis.items():
            self.app.add_api_route(path, self.endpoint(api), methods=["POST"], response_model=None)
        self.app.add_api_route("/sessions/{sid}", self.get_session, methods=["GET"], response_class=Response)
        self.app.add_api_route(
            "/sessions/{sid}", self.delete_session, methods=["DELETE"], status_code=HTTPStatus.NO_CONTENT
        )
        for kind in (RequestValidationError, GatewayError, ChatTemplateError, InferenceEngineError):
            self.app.add_exception_handler(kind, self.refuse)

    @asynccontextmanager
    async def lifespan(self, _: FastAPI) -> AsyncIterator[None]:
        try:
            async with asyncio.timeout(self.ready_timeout):
                await self.gateway.engine.ready()
        except TimeoutError as e:
            raise InferenceEngineError(f"the engine could not generate within {self.ready_timeout:.0f} s") from e
        yield

    def _session(self, sid: str) -> Session:
        if (session := self.sessions.get(sid)) is None:
            raise UnknownSessionError(f"no session {sid!r}; open one with POST /sessions")
        return session

    async def health(self) -> dict[str, str]:
        """Answered only once startup found the engine able to generate."""
        return {"status": "ok"}

    async def create_session(self, request: SessionRequest | None = None) -> dict[str, str]:
        request = request or SessionRequest()
        self.gateway.refuse_reserved(request.params)
        session = Session(params=request.params, resume_aborted=request.resume_aborted)
        self.sessions[session.id] = session
        return {"id": session.id}

    async def get_session(self, sid: str) -> Response:
        """The session as `Session.pack` writes it, for a trainer to collect."""
        return Response(self._session(sid).pack(), media_type="application/octet-stream")

    async def delete_session(self, sid: str) -> None:
        """Forget the session; a turn still running finishes, but on a session no longer listed."""
        del self.sessions[self._session(sid).id]

    def _status(self, http: Request, exc: Exception) -> HTTPStatus:
        """The status for `exc`, logged with its reason, which the access log does not show."""
        match exc:
            case GatewayError():
                status = exc.status_code
            case InferenceEngineError():
                status = HTTPStatus.BAD_GATEWAY
            case _:
                status = HTTPStatus.BAD_REQUEST
        logger.warning("%s %s answered %d: %s", http.method, http.url.path, status, exc)
        return status

    async def refuse(self, http: Request, exc: Exception) -> JSONResponse:
        # A refusal answers in its route's API; a route of none, such as `/sessions`, in chat's.
        api = self.apis.get(getattr(http.scope.get("route"), "path", ""), OpenAIChatRequest)
        if isinstance(exc, RequestValidationError):
            # FastAPI locates body errors under `body`, and the body is the harness's whole request.
            errors = [{**e, "loc": e["loc"][1:]} for e in exc.errors()]
            exc = ValidationError.from_exception_data(api.__name__, cast(list[InitErrorDetails], errors))
        status = self._status(http, exc)
        return JSONResponse(api.error(exc, status), status)

    async def _stream(
        self, turn: asyncio.Future[ChatResponse], request: HarnessRequest, http: Request
    ) -> AsyncIterator[str]:
        """The events of a turn still generating: its API's opening, keep-alives until it ends, then its reply or error.

        Its headers are sent already, so an error reaches the harness as its API's error events, not a status. A
        harness hanging up cancels this stream, and the turn with it.
        """
        try:
            for event in request.start_events():
                yield event
            while not turn.done():
                await asyncio.wait({turn}, timeout=self.keepalive)
                if not turn.done():
                    yield request.keepalive_event()
            try:
                reply = turn.result()
            except (GatewayError, ChatTemplateError, InferenceEngineError) as exc:
                for event in request.error_events(exc, self._status(http, exc)):
                    yield event
                return
            for event in request.response_events(reply):
                yield event
        finally:
            turn.cancel()

    async def serve(self, sid: str, request: HarnessRequest, http: Request) -> BaseModel | Response:
        """The turn's response in the request's API, as one response or, when asked, a stream.

        A harness hanging up first cancels the turn; the 499 sent back is never read. A stream opens once the turn
        starts generating and sends keep-alives until it is done: clients time out waiting for a response's headers
        or on a silent stream (OpenAI's SDKs at 600 s, opencode and Codex at 300 s), and a generation can run longer.
        """

        async def hung_up() -> None:
            # Past the body, ASGI sends only `http.disconnect`; anything else is passed over.
            while (await http.receive())["type"] != "http.disconnect":
                pass

        generating = asyncio.Event()
        turn = asyncio.create_task(self.gateway.chat(self._session(sid), request.chat_request(), generating))
        # A stream opens once the turn generates, a response once the turn ends, unless the harness hangs up first.
        opened = asyncio.create_task(generating.wait()) if request.stream else turn
        gone = asyncio.create_task(hung_up())
        await asyncio.wait({turn, opened, gone}, return_when=asyncio.FIRST_COMPLETED)
        gone.cancel()
        opened.cancel()
        if not turn.done():
            if gone.done():
                turn.cancel()
                return Response(status_code=499)
            return StreamingResponse(self._stream(turn, request, http), media_type="text/event-stream")
        # The turn ended before any reply was sent, so a refusal raises here and keeps its status.
        reply = turn.result()
        if request.stream:
            events = [*request.start_events(), *request.response_events(reply)]
            return StreamingResponse(events, media_type="text/event-stream")
        return request.response(reply)

    def endpoint(self, api: type[HarnessRequest]) -> Callable[..., Awaitable[BaseModel | Response]]:
        """`serve` for one API's route; FastAPI parses the body as the type its parameter is annotated with."""

        async def endpoint(sid: str, request: HarnessRequest, http: Request) -> BaseModel | Response:
            return await self.serve(sid, request, http)

        endpoint.__annotations__["request"] = api
        return endpoint
