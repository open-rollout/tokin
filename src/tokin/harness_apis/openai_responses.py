from __future__ import annotations

import json
from http import HTTPStatus
from typing import Annotated, Any, Literal, NotRequired, Self, TypedDict, cast

from openai.types.chat import ChatCompletionMessageFunctionToolCall
from openai.types.completion_usage import CompletionUsage
from openai.types.responses import Response
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    ValidationError,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_validator,
    model_validator,
    with_config,
)

from ..gateway import ChatRequest, ChatResponse, ContextLengthExceededError
from .openai_chat import OpenAIChatRequest
from .request import HarnessRequest

# A response before its turn is recorded, which is what gives it its id.
_IN_PROGRESS = {"object": "response", "status": "in_progress"}


# Items and parts refuse keys they do not declare. Those an item is served with come back and are read by nothing:
# no template renders them, and reasoning continues by its raw text.
class TextPart(BaseModel, extra="forbid"):
    type: Literal["input_text", "output_text"]
    text: str
    annotations: list[dict[str, Any]] | None = None
    logprobs: list[dict[str, Any]] | None = None


# Templates render a message's content as a string, so parts are read as their text joined.
type Text = Annotated[
    str | list[TextPart], AfterValidator(lambda c: c if isinstance(c, str) else "".join(p.text for p in c))
]
type Status = Literal["in_progress", "completed", "incomplete"]


class ServedItem(BaseModel, extra="forbid"):
    id: str | None = None


class MessageItem(ServedItem):
    type: Literal["message"]
    role: Literal["user", "assistant", "system", "developer"]
    content: Text
    status: Status | None = None
    phase: Literal["commentary", "final_answer"] | None = None


class ReasoningText(BaseModel, extra="forbid"):
    type: Literal["reasoning_text"]
    text: str


class ReasoningItem(ServedItem):
    type: Literal["reasoning"]
    content: list[ReasoningText] | None = None
    summary: list[dict[str, Any]] | None = None
    encrypted_content: str | None = None
    status: Status | None = None

    @model_validator(mode="after")
    def has_raw_text(self) -> Self:
        if self.content is None:
            raise ValueError(
                "a reasoning item must carry its raw text in `content`: a session continues a turn by the reasoning "
                "the model sampled, which `summary` and `encrypted_content` do not give back"
            )
        return self


class FunctionCallItem(ServedItem):
    type: Literal["function_call"]
    call_id: str
    name: str
    namespace: str | None = None
    arguments: str
    status: Status | None = None
    # Null only, as a response not streamed serves it: every call tokin serves is the model's own.
    caller: None = None


class FunctionCallOutputItem(ServedItem):
    type: Literal["function_call_output"]
    call_id: str
    output: Text


class CustomToolCallItem(ServedItem):
    type: Literal["custom_tool_call"]
    call_id: str
    name: str
    namespace: str | None = None
    input: str
    caller: None = None

    @field_validator("namespace")
    @classmethod
    def no_namespace(cls, value: str | None) -> str | None:
        if value is not None:
            raise ValueError("a custom tool is not served in a namespace: a namespace's tools are served as functions")
        return value


class CustomToolCallOutputItem(ServedItem):
    type: Literal["custom_tool_call_output"]
    call_id: str
    output: Text


type Item = Annotated[
    MessageItem
    | ReasoningItem
    | FunctionCallItem
    | FunctionCallOutputItem
    | CustomToolCallItem
    | CustomToolCallOutputItem,
    Field(discriminator="type"),
]


@with_config(ConfigDict(extra="forbid"))
class FunctionTool(TypedDict):
    """Chat's function definition, its keys at the tool's top level."""

    type: Literal["function"]
    name: str
    description: NotRequired[str | None]
    parameters: NotRequired[dict[str, Any] | None]
    strict: NotRequired[bool | None]


@with_config(ConfigDict(extra="forbid"))
class TextFormat(TypedDict):
    type: Literal["text"]


@with_config(ConfigDict(extra="forbid"))
class GrammarFormat(TypedDict):
    type: Literal["grammar"]
    syntax: Literal["lark", "regex"]
    definition: str


@with_config(ConfigDict(extra="forbid"))
class CustomTool(TypedDict):
    """A tool called with free text, which `format` describes."""

    type: Literal["custom"]
    name: str
    description: NotRequired[str]
    format: NotRequired[Annotated[TextFormat | GrammarFormat, Field(discriminator="type")]]


@with_config(ConfigDict(extra="forbid"))
class NamespaceTool(TypedDict):
    """Functions grouped under `name`, which a call to one of them carries as its `namespace`."""

    type: Literal["namespace"]
    name: str
    description: str
    tools: list[FunctionTool]


type Tool = Annotated[FunctionTool | CustomTool | NamespaceTool, Field(discriminator="type")]


class Reasoning(BaseModel, extra="forbid"):
    """Empty: an effort or a summary would change the turn, and the template takes neither."""


class OpenAIResponsesRequest(HarnessRequest, extra="forbid"):
    """The harness's `/v1/responses` body, served as the chat request it translates to.

    Undeclared keys are refused by name, and each declared one takes only the values `tokin` serves, so a refusal is
    located at this body's keys. Leading system and developer messages become one system message: Qwen templates
    refuse a `developer` role and a second system message.
    """

    path = "/responses"

    model: str
    input: list[Item]
    tools: list[Tool] | None = None
    tool_choice: Literal["auto"] | None = None
    store: Literal[False] | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_output_tokens: int | None = Field(None, ge=1)
    parallel_tool_calls: bool | None = None
    reasoning: Reasoning | None = None

    # Sent by Codex and read by nothing: none changes the turn, and reasoning comes back as raw text, not encrypted.
    include: list[Literal["reasoning.encrypted_content"]] | None = None
    prompt_cache_key: str | None = None
    client_metadata: dict[str, Any] | None = None

    _chat: OpenAIChatRequest = PrivateAttr()

    @staticmethod
    def _flatten(tools: list[Tool]) -> dict[str, tuple[NamespaceTool | None, FunctionTool | CustomTool]]:
        """Each tool by the name chat serves it under, with its namespace; a namespace's functions are `namespace.name`.

        A name served twice is refused at the second, since a chat call names its tool alone. OpenAI's names hold only
        letters, digits, `_` and `-`, so only a name OpenAI refuses can collide with a namespaced one.
        """
        flat: dict[str, tuple[NamespaceTool | None, FunctionTool | CustomTool]] = {}
        for i, tool in enumerate(tools):
            members: list[tuple[NamespaceTool | None, FunctionTool | CustomTool, tuple[int | str, ...]]]
            if tool["type"] == "namespace":
                members = [(tool, f, (i, "namespace", "tools", j)) for j, f in enumerate(tool["tools"])]
            else:
                members = [(None, tool, (i, tool["type"]))]
            for namespace, member, loc in members:
                name = member["name"] if namespace is None else f"{namespace['name']}.{member['name']}"
                if name in flat:
                    error = ValueError(f"`{name}` is the name of an earlier tool too: a chat call names its tool alone")
                    raise ValidationError.from_exception_data(
                        "tools",
                        [{"type": "value_error", "loc": (*loc, "name"), "input": name, "ctx": {"error": error}}],
                    )
                flat[name] = (namespace, member)
        return flat

    @field_validator("tools", mode="wrap")
    @classmethod
    def validate_tools(cls, value: Any, handler: ValidatorFunctionWrapHandler) -> Any:
        """Validated, but kept as sent, so that chat's tools keep the order sent, which templates print."""
        handler(value)
        cls._flatten(value or [])
        return value

    @field_validator("parallel_tool_calls")
    @classmethod
    def no_false_with_tools(cls, value: bool | None, info: ValidationInfo) -> bool | None:
        if value is False and info.data.get("tools"):
            raise ValueError(
                "`false` is not served with tools: generation is unconstrained, so a turn may make several calls"
            )
        return value

    @model_validator(mode="after")
    def translate(self) -> Self:
        """Map this body onto its chat request, which chat validates as a check: refusing a valid body is a bug.

        The bug is raised as a `RuntimeError`, which `pydantic` passes on uncaught, so it is answered with 500: a 400
        would ask the harness to fix a body that is fine.
        """
        system: list[str] = []
        messages: list[dict[str, Any]] = []
        # A turn's items come in the order `response` gives them: reasoning, then text, then calls, all one reply.
        for item in self.input:
            reply = messages[-1] if messages and messages[-1]["role"] == "assistant" else None
            match item:
                case MessageItem(role="system" | "developer") if not messages:
                    system.append(cast(str, item.content))
                case MessageItem(role="assistant") if reply and not (reply["content"] or reply.get("tool_calls")):
                    reply["content"] = item.content
                case MessageItem():
                    messages.append({"role": item.role, "content": item.content})
                case ReasoningItem():
                    text = "".join(part.text for part in item.content or [])
                    messages.append({"role": "assistant", "content": None, "reasoning_content": text})
                case FunctionCallItem() | CustomToolCallItem():
                    if reply is None:
                        messages.append(reply := {"role": "assistant", "content": None})
                    function: dict[str, str]
                    if isinstance(item, CustomToolCallItem):
                        function = {"name": item.name, "arguments": json.dumps({"input": item.input})}
                    else:
                        name = item.name if item.namespace is None else f"{item.namespace}.{item.name}"
                        function = {"name": name, "arguments": item.arguments}
                    reply.setdefault("tool_calls", []).append(
                        {"id": item.call_id, "type": "function", "function": function}
                    )
                case FunctionCallOutputItem() | CustomToolCallOutputItem():
                    messages.append({"role": "tool", "tool_call_id": item.call_id, "content": item.output})
        if system:
            messages.insert(0, {"role": "system", "content": "\n\n".join(system)})
        chat = self.model_dump(include={"model", "tool_choice", "store", "temperature", "top_p"}, exclude_unset=True)
        chat["messages"] = messages
        if self.tools is not None:
            chat["tools"] = []
            for name, (namespace, tool) in self._flatten(self.tools).items():
                schema: dict[str, Any]
                if tool["type"] == "custom":
                    parameter: dict[str, Any] = {"type": "string"}
                    # A grammar is shown to the model, not enforced: generation is unconstrained.
                    if (form := tool.get("format")) and form["type"] == "grammar":
                        parameter["description"] = f"Must match this {form['syntax']} grammar:\n{form['definition']}"
                    schema = {k: v for k, v in tool.items() if k in ("name", "description")}
                    schema["parameters"] = {"type": "object", "properties": {"input": parameter}, "required": ["input"]}
                else:
                    # Chat takes no null `description` or `parameters`: a null there is the key left out.
                    schema = {k: v for k, v in tool.items() if k != "type" and v is not None} | {"name": name}
                    # Chat has no namespace to describe, so its description opens each of its functions'.
                    if namespace and namespace["description"]:
                        described = (namespace["description"], tool.get("description"))
                        schema["description"] = "\n\n".join(filter(None, described))
                chat["tools"].append({"type": "function", "function": schema})
        if "max_output_tokens" in self.model_fields_set:
            chat["max_completion_tokens"] = self.max_output_tokens
        # Without tools it constrains nothing, and Codex's compaction sends `false` with `tools: []`.
        if self.tools and "parallel_tool_calls" in self.model_fields_set:
            chat["parallel_tool_calls"] = self.parallel_tool_calls
        try:
            self._chat = OpenAIChatRequest.model_validate(chat)
        except ValidationError as e:
            raise RuntimeError(f"a bug in tokin: chat refuses the chat request this valid body maps to: {e}") from e
        return self

    @staticmethod
    def _event(kind: str, **data: Any) -> str:
        return f"event: {kind}\ndata: {json.dumps({'type': kind, **data})}\n\n"

    def chat_request(self) -> ChatRequest:
        return self._chat.chat_request()

    def start_events(self) -> list[str]:
        return [self._event("response.created", response=_IN_PROGRESS)]

    def keepalive_event(self) -> str:
        """A real event, not a comment: Codex resets its idle timeout on events only."""
        return self._event("response.in_progress", response=_IN_PROGRESS)

    def response(self, reply: ChatResponse) -> Response:
        """The `Response` for `reply`, from chat's completion; a cut at `max_output_tokens` is `incomplete`.

        Built as the SDK reads OpenAI's, unvalidated: a valid `Response` requires usage details and an echo of the
        request's tool settings, which `tokin` does not report.
        """
        completion = self._chat.response(reply)
        message, usage = completion.choices[0].message, cast(CompletionUsage, completion.usage)
        status = "incomplete" if completion.choices[0].finish_reason == "length" else "completed"
        output: list[dict[str, Any]] = []
        if (reasoning := (message.model_extra or {}).get("reasoning_content")) is not None:
            content = [{"type": "reasoning_text", "text": reasoning}]
            output.append(
                {
                    "type": "reasoning",
                    "id": f"rs_{completion.id}",
                    "summary": [],
                    "content": content,
                    "encrypted_content": None,
                }
            )
        # A cut turn's calls are left out: they are unfinished, and a harness runs a call once its item is done.
        calls = [] if status == "incomplete" else message.tool_calls or []
        # A reply of nothing else is still a message: left out, the next request would not continue it.
        if message.content or not (output or calls):
            text = {"type": "output_text", "text": message.content or "", "annotations": []}
            output.append(
                {
                    "type": "message",
                    "id": f"msg_{completion.id}",
                    "role": "assistant",
                    "status": status,
                    "content": [text],
                }
            )
        tools = self._flatten(self.tools or [])
        for call in calls:
            function = cast(ChatCompletionMessageFunctionToolCall, call).function
            match tools.get(function.name):
                case (_, {"type": "custom"}):
                    try:
                        given = json.loads(function.arguments)
                    except ValueError:
                        given = None
                    # Arguments not `{"input": <text>}` go as read, a block unread as its text, as a function's do.
                    if not (isinstance(given, dict) and given.keys() == {"input"} and isinstance(given["input"], str)):
                        given = {"input": function.arguments}
                    output.append({"type": "custom_tool_call", "call_id": call.id, "name": function.name, **given})
                    continue
                case ({"name": namespace}, {"name": name}):
                    # Codex routes a call by its namespace and the name within it.
                    named = {"namespace": namespace, "name": name}
                case _:
                    named = {"name": function.name}
            output.append({"type": "function_call", "call_id": call.id, **named, "arguments": function.arguments})
        return Response.construct(
            id=f"resp_{completion.id}",
            object="response",
            created_at=completion.created,
            model=completion.model,
            status=status,
            incomplete_details={"reason": "max_output_tokens"} if status == "incomplete" else None,
            output=output,
            usage={
                "input_tokens": usage.prompt_tokens,
                "output_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            },
        )

    def response_events(self, reply: ChatResponse) -> list[str]:
        """The response as OpenAI's stream, past `start_events`, in one burst: each item done, then how it ended.

        One burst, as the turn is done before it is sent. Done is the event that enters an item into Codex's history.
        """
        response = self.response(reply).model_dump(mode="json", exclude_unset=True)
        done = [
            self._event("response.output_item.done", output_index=i, item=item)
            for i, item in enumerate(response["output"])
        ]
        return [*done, self._event(f"response.{response['status']}", response=response)]

    @classmethod
    def error(cls, exc: Exception, status: HTTPStatus) -> dict[str, Any]:
        """Chat's error body, OpenAI's for both APIs, with a context overflow located in `input`."""
        body = OpenAIChatRequest.error(exc, status)
        if isinstance(exc, ContextLengthExceededError):
            body["error"]["param"] = "input"
        return body

    def error_events(self, exc: Exception, status: HTTPStatus) -> list[str]:
        """Failed, with the error's code and message, which Codex branches on."""
        error = {key: self.error(exc, status)["error"][key] for key in ("code", "message")}
        return [self._event("response.failed", response={"object": "response", "status": "failed", "error": error})]
