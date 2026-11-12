import json
from http import HTTPStatus

import pytest
from conftest import said
from fastapi.encoders import jsonable_encoder
from openai.types.responses import (
    CustomToolParam,
    EasyInputMessageParam,
    FunctionToolParam,
    NamespaceToolParam,
    ResponseCustomToolCallOutputParam,
    ResponseCustomToolCallParam,
    ResponseFunctionToolCallParam,
    ResponseInputTextParam,
    ResponseOutputMessageParam,
    ResponseOutputTextParam,
    ResponseReasoningItemParam,
)
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from openai.types.responses.response_input_item_param import FunctionCallOutput
from openai.types.responses.response_reasoning_item_param import Content
from openai.types.shared_params.custom_tool_input_format import Grammar, Text
from pydantic import ValidationError

from tokin.gateway import ChatResponse, ContextLengthExceededError
from tokin.harness_apis import OpenAIChatRequest, OpenAIResponsesRequest, openai_responses as responses
from tokin.rollout import FinishReason, Generation
from tokin.session import Session

# OpenAI's keys `tokin` does not serve: each is declared when a harness is seen sending it.
UNSERVED = {
    "background",
    "context_management",
    "conversation",
    "instructions",
    "max_tool_calls",
    "metadata",
    "moderation",
    "previous_response_id",
    "prompt",
    "prompt_cache_options",
    "prompt_cache_retention",
    "safety_identifier",
    "service_tier",
    "stream_options",
    "text",
    "top_logprobs",
    "truncation",
    "user",
}

# Keys outside OpenAI's that a harness is seen sending: Codex its telemetry.
EXTRA = {"client_metadata"}

USER = {"type": "message", "role": "user", "content": "hi"}
TOOL = {"type": "function", "name": "f", "description": "d", "strict": False, "parameters": {"type": "object"}}
CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}
GRAMMAR = {"type": "grammar", "syntax": "lark", "definition": "start: x"}
PATCH = {"type": "custom", "name": "apply_patch", "description": "d", "format": GRAMMAR}
CLOCK = {"type": "namespace", "name": "clock", "description": "Time.", "tools": [{**TOOL, "name": "sleep"}]}
PATCH_CALL = {"id": "c1", "type": "function", "function": {"name": "apply_patch", "arguments": '{"input": "p"}'}}


def request(**over):
    return OpenAIResponsesRequest.model_validate({"model": "m", "input": [USER], **over})


def reply(message, finish=FinishReason.STOP):
    return ChatResponse(message, Generation([1, 2], finish), 3, "chatcmpl-1")


def test_every_openai_key_is_served_or_listed_unserved():
    # A bump of `openai` that adds a key fails here, so the new key gets a decision rather than a default.
    upstream = ResponseCreateParamsStreaming.__required_keys__ | ResponseCreateParamsStreaming.__optional_keys__
    served = set(OpenAIResponsesRequest.model_fields) - EXTRA
    assert served <= upstream
    assert upstream - served == UNSERVED


# Each of `tokin`'s Responses types, and OpenAI's it declares a part of.
DECLARED = {
    responses.TextPart: (ResponseInputTextParam, ResponseOutputTextParam),
    responses.MessageItem: (EasyInputMessageParam, ResponseOutputMessageParam),
    responses.ReasoningText: Content,
    responses.ReasoningItem: ResponseReasoningItemParam,
    responses.FunctionCallItem: ResponseFunctionToolCallParam,
    responses.FunctionCallOutputItem: FunctionCallOutput,
    responses.CustomToolCallItem: ResponseCustomToolCallParam,
    responses.CustomToolCallOutputItem: ResponseCustomToolCallOutputParam,
    responses.FunctionTool: FunctionToolParam,
    responses.TextFormat: Text,
    responses.GrammarFormat: Grammar,
    responses.CustomTool: CustomToolParam,
    responses.NamespaceTool: NamespaceToolParam,
}


@pytest.mark.parametrize(("declared", "upstream"), DECLARED.items(), ids=[t.__name__ for t in DECLARED])
def test_declared_keys_are_openais(declared, upstream):
    keys = set().union(*(t.__annotations__ for t in (upstream if isinstance(upstream, tuple) else (upstream,))))
    assert set(getattr(declared, "model_fields", None) or declared.__annotations__) <= keys


def test_chat_request_params_leave_out_what_the_harness_did_not_set():
    assert request(temperature=0.7).chat_request().params == {"temperature": 0.7}


def test_max_output_tokens_is_max_tokens():
    assert request(max_output_tokens=5).chat_request().params == {"max_tokens": 5}


def test_parallel_tool_calls_without_tools_is_dropped():
    assert request(tools=[], parallel_tool_calls=False).chat_request().tools == []


def test_tools_become_chat_functions_in_the_order_sent():
    (tool,) = request(tools=[TOOL]).chat_request().tools
    assert list(tool) == ["type", "function"] and list(tool["function"]) == list(TOOL)[1:]


def test_a_tools_nulls_are_left_out_of_its_function():
    (tool,) = request(tools=[{**TOOL, "description": None, "parameters": None}]).chat_request().tools
    assert tool["function"] == {"name": "f", "strict": False}


def test_a_custom_tool_is_a_function_of_one_string_input_its_grammar_describes():
    (tool,) = request(tools=[PATCH]).chat_request().tools
    text = {"type": "string", "description": "Must match this lark grammar:\nstart: x"}
    assert tool["function"] == {
        "name": "apply_patch",
        "description": "d",
        "parameters": {"type": "object", "properties": {"input": text}, "required": ["input"]},
    }


def test_a_namespaces_functions_are_named_and_described_within_it():
    (tool,) = request(tools=[CLOCK]).chat_request().tools
    assert (tool["function"]["name"], tool["function"]["description"]) == ("clock.sleep", "Time.\n\nd")


def test_leading_system_and_developer_messages_become_one_system_message():
    system = {"type": "message", "role": "system", "content": "a"}
    developer = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "b"}]}
    assert request(input=[system, developer, USER]).chat_request().messages[:2] == [
        {"role": "system", "content": "a\n\nb"},
        {"role": "user", "content": "hi"},
    ]


@pytest.mark.parametrize("role", ["developer", "system"])
def test_a_later_developer_or_system_message_is_a_user_message_tagged_with_its_role(role):
    later = {"type": "message", "role": role, "content": "b"}
    assert request(input=[USER, later]).chat_request().messages[1:] == [
        {"role": "user", "content": f"<{role}_message>\nb\n</{role}_message>"}
    ]


def test_text_parts_are_joined():
    parts = [{"type": "input_text", "text": "h"}, {"type": "input_text", "text": "i"}]
    assert request(input=[{**USER, "content": parts}]).chat_request().messages == [{"role": "user", "content": "hi"}]


def test_function_calls_join_the_reply_and_outputs_answer_them():
    said = {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": "ok"}]}
    call = {"type": "function_call", "call_id": "c1", "name": "f", "arguments": "{}"}
    output = {"type": "function_call_output", "call_id": "c1", "output": "done"}
    assert request(input=[USER, said, call, output]).chat_request().messages[1:] == [
        {"role": "assistant", "content": "ok", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]


def test_custom_tool_calls_are_calls_of_their_input_and_outputs_answer_them():
    call = {"type": "custom_tool_call", "call_id": "c1", "name": "apply_patch", "input": "p"}
    output = {"type": "custom_tool_call_output", "call_id": "c1", "output": "done"}
    assert request(input=[USER, call, output]).chat_request().messages[1:] == [
        {"role": "assistant", "content": None, "tool_calls": [PATCH_CALL]},
        {"role": "tool", "tool_call_id": "c1", "content": "done"},
    ]


def test_a_namespaced_function_call_is_called_by_its_name_within_the_namespace():
    call = {"type": "function_call", "call_id": "c1", "namespace": "clock", "name": "sleep", "arguments": "{}"}
    (message,) = request(input=[USER, call]).chat_request().messages[1:]
    assert message["tool_calls"][0]["function"]["name"] == "clock.sleep"


def test_a_reasoning_item_becomes_the_reasoning_of_its_reply():
    thought = {"type": "reasoning", "summary": [], "content": [{"type": "reasoning_text", "text": "hm"}]}
    answer = {"type": "message", "role": "assistant", "content": "ok"}
    assert request(input=[USER, thought, answer]).chat_request().messages[1:] == [
        {"role": "assistant", "content": "ok", "reasoning_content": "hm"}
    ]


async def test_a_reply_with_reasoning_sent_back_continues_its_turn(gateway):
    g, session = gateway(said("<think>hm</think>ok"), said("done")), Session()
    first = request()
    output = first.response(await g.chat(session, first.chat_request())).model_dump(exclude_unset=True)["output"]
    await g.chat(session, request(input=[USER, *output, {**USER, "content": "again"}]).chat_request())
    assert output[0]["type"] == "reasoning" and session.turns[1].parent == 0


async def test_custom_and_namespaced_calls_sent_back_continue_their_turn(gateway):
    calls = [{"name": "apply_patch", "arguments": {"input": "p"}}, {"name": "clock.sleep", "arguments": {}}]
    text = "".join(f"<tool_call>{json.dumps(c)}</tool_call>" for c in calls)
    g, session = gateway(said(text), said("done")), Session()
    first = request(tools=[PATCH, CLOCK])
    served = await g.chat(session, first.chat_request())
    output = first.response(served).model_dump(exclude_unset=True)["output"]
    results = [{"type": f"{item['type']}_output", "call_id": item["call_id"], "output": "ok"} for item in output]
    again = request(tools=[PATCH, CLOCK], input=[USER, *output, *results]).chat_request()
    await g.chat(session, again)
    assert again.messages[1]["tool_calls"] == served.message["tool_calls"] and session.turns[1].parent == 0


def test_a_response_not_streamed_is_taken_back_as_served():
    calls = [PATCH_CALL, {**CALL, "id": "c2", "function": {"name": "clock.sleep", "arguments": "{}"}}]
    message = {"role": "assistant", "content": "ok", "reasoning_content": "hm", "tool_calls": calls}
    served = jsonable_encoder(request(tools=[PATCH, CLOCK]).response(reply(message)))["output"]
    again = request(tools=[PATCH, CLOCK], input=[USER, *served]).chat_request()
    assert again.messages[1] == message


def test_codexs_requests_become_chat_requests():
    # Codex 0.167.1's first two requests of a turn with one call, its texts cut short.
    prompt = [
        {
            "type": "message",
            "id": "msg_1",
            "role": "developer",
            "content": [{"type": "input_text", "text": "You are a coding agent"}],
        },
        {
            "type": "message",
            "id": "msg_2",
            "role": "developer",
            "content": [{"type": "input_text", "text": "<skills>"}, {"type": "input_text", "text": "<permissions>"}],
        },
        {
            "type": "message",
            "id": "msg_3",
            "role": "user",
            "content": [{"type": "input_text", "text": "<environment_context>"}],
        },
        {"type": "message", "id": "msg_4", "role": "user", "content": [{"type": "input_text", "text": "x"}]},
    ]
    turn = [
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [],
            "content": [{"type": "reasoning_text", "text": "thinking"}],
            "encrypted_content": None,
        },
        {"type": "function_call", "id": "fc_1", "name": "exec_command", "arguments": '{"cmd": "ls"}', "call_id": "c1"},
        {"type": "function_call_output", "id": "fco_1", "call_id": "c1", "output": "a.txt"},
    ]
    body = {
        "model": "m",
        "stream": True,
        "tools": [{"type": "function", "name": "exec_command", "description": "d", "strict": False, "parameters": {}}],
        "tool_choice": "auto",
        "parallel_tool_calls": True,
        "reasoning": {},
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "prompt_cache_key": "k",
        "client_metadata": {"turn_id": "t", "session_id": "s"},
    }
    roles = [
        [m["role"] for m in OpenAIResponsesRequest.model_validate({**body, "input": i}).chat_request().messages]
        for i in (prompt, prompt + turn)
    ]
    assert roles == [["system", "user", "user"], ["system", "user", "user", "assistant", "tool"]]


def test_response_carries_text_and_calls_with_their_ids():
    output = request().response(reply({"role": "assistant", "content": "ok", "tool_calls": [CALL]})).output
    assert [item.type for item in output] == ["message", "function_call"]
    assert output[0].content[0].text == "ok" and output[1].call_id == "c1"


def test_response_gives_a_custom_tools_call_its_input():
    (item,) = request(tools=[PATCH]).response(reply({"role": "assistant", "tool_calls": [PATCH_CALL]})).output
    assert (item.type, item.call_id, item.name, item.input) == ("custom_tool_call", "c1", "apply_patch", "p")


@pytest.mark.parametrize("arguments", ['{"input": "p", "x": 1}', '{"input": 1}', "<tool_call>apply_patch</tool_call>"])
def test_response_gives_a_custom_tools_other_arguments_as_written(arguments):
    call = {**PATCH_CALL, "function": {"name": "apply_patch", "arguments": arguments}}
    (item,) = request(tools=[PATCH]).response(reply({"role": "assistant", "tool_calls": [call]})).output
    assert item.input == arguments


def test_response_gives_a_namespaced_call_its_namespace():
    call = {**CALL, "function": {"name": "clock.sleep", "arguments": "{}"}}
    (item,) = request(tools=[CLOCK]).response(reply({"role": "assistant", "tool_calls": [call]})).output
    assert (item.type, item.namespace, item.name) == ("function_call", "clock", "sleep")


def test_response_with_neither_text_nor_calls_is_an_empty_message():
    (item,) = request().response(reply({"role": "assistant", "content": None})).output
    assert item.type == "message" and item.content[0].text == ""


def test_response_usage_counts_input_and_output_tokens():
    usage = request().response(reply({"role": "assistant", "content": "ok"})).usage
    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (3, 2, 5)


def test_response_id_is_the_turns():
    assert request().response(reply({"role": "assistant", "content": "ok"})).id == "resp_chatcmpl-1"


def test_response_cut_at_max_output_tokens_is_incomplete():
    response = request().response(reply({"role": "assistant", "content": "o"}, FinishReason.LENGTH))
    assert response.status == "incomplete" and response.incomplete_details.reason == "max_output_tokens"


def test_response_cut_leaves_its_calls_out():
    response = request().response(
        reply({"role": "assistant", "content": "o", "tool_calls": [CALL]}, FinishReason.LENGTH)
    )
    assert [item.type for item in response.output] == ["message"]


def test_response_events_send_each_item_done_then_the_end():
    events = request().response_events(reply({"role": "assistant", "content": "ok", "tool_calls": [CALL]}))
    kinds = [event.split("\n")[0] for event in events]
    assert kinds == ["event: response.output_item.done"] * 2 + ["event: response.completed"]


def test_error_events_carry_a_context_overflows_code():
    (event,) = request().error_events(ContextLengthExceededError(9, 5), HTTPStatus.BAD_REQUEST)
    data = json.loads(event.split("data: ")[1])
    assert data["type"] == "response.failed" and data["response"]["error"]["code"] == "context_length_exceeded"


def test_error_locates_a_context_overflow_in_input():
    error = OpenAIResponsesRequest.error(ContextLengthExceededError(9, 5), HTTPStatus.BAD_REQUEST)["error"]
    assert error["code"] == "context_length_exceeded" and error["param"] == "input"


def test_undeclared_key_is_refused_by_name():
    with pytest.raises(ValidationError, match="truncation"):
        request(truncation="auto")


def test_a_reasoning_effort_is_refused():
    with pytest.raises(ValidationError, match="reasoning.effort"):
        request(reasoning={"effort": "high"})


def test_a_reasoning_item_without_its_raw_text_is_refused():
    with pytest.raises(ValidationError, match="must carry its raw text"):
        request(input=[USER, {"type": "reasoning", "summary": [], "encrypted_content": "x"}])


def test_an_undeclared_key_in_a_tool_is_refused_by_name():
    with pytest.raises(ValidationError, match="tools.0.function.output_schema"):
        request(tools=[{**TOOL, "output_schema": {}}])


def test_a_tool_of_another_type_is_refused():
    with pytest.raises(ValidationError, match="tools.0\n  Value error, `web_search` is not served: tokin runs no tool"):
        request(tools=[{"type": "web_search"}])


def test_an_item_of_another_type_is_refused():
    with pytest.raises(
        ValidationError, match="input.1\n  Value error, `web_search_call` is not served: tokin runs no tool"
    ):
        request(input=[USER, {"type": "web_search_call", "id": "ws_1", "status": "completed"}])


IMAGE = {"type": "input_image", "image_url": "data:image/png;base64,", "detail": "auto"}


@pytest.mark.parametrize(
    ("item", "loc"),
    [
        ({**USER, "content": [IMAGE]}, "message.content.0"),
        ({**USER, "content": [{"type": "input_file", "file_id": "f"}]}, "message.content.0"),
        ({"type": "function_call_output", "call_id": "c1", "output": [IMAGE]}, "function_call_output.output.0"),
    ],
    ids=["image", "file", "image output"],
)
def test_a_part_other_than_text_is_refused(item, loc):
    with pytest.raises(
        ValidationError, match=f"input.1.{loc}\n  Value error, `input_.*` is not served: .*images are not supported yet"
    ):
        request(input=[USER, item])


def test_an_undeclared_key_in_an_item_is_refused_by_name():
    with pytest.raises(ValidationError, match="input.0.message.encrypted_content"):
        request(input=[{**USER, "encrypted_content": "e"}])


def test_a_custom_tool_call_in_a_namespace_is_refused():
    call = {"type": "custom_tool_call", "call_id": "c1", "namespace": "n", "name": "apply_patch", "input": "p"}
    with pytest.raises(ValidationError, match="input.1.custom_tool_call.namespace\n.*not served in a namespace"):
        request(input=[USER, call])


def test_a_name_two_tools_are_served_under_is_refused_at_the_second():
    with pytest.raises(ValidationError) as e:
        request(tools=[{**TOOL, "name": "clock.sleep"}, CLOCK])
    (error,) = e.value.errors()
    assert error["loc"] == ("tools", 1, "namespace", "tools", 0, "name") and "an earlier tool" in error["msg"]


def test_parallel_tool_calls_false_with_tools_is_refused():
    with pytest.raises(ValidationError, match="parallel_tool_calls\n.*not served with tools"):
        request(tools=[TOOL], parallel_tool_calls=False)


@pytest.mark.parametrize(
    ("key", "value"), [("store", True), ("tool_choice", "required"), ("max_output_tokens", 0), ("model", None)]
)
def test_a_value_not_served_is_refused_at_its_key(key, value):
    with pytest.raises(ValidationError) as e:
        request(**{key: value})
    assert [err["loc"] for err in e.value.errors()] == [(key,)]


def test_a_tool_without_a_name_is_refused_at_it():
    with pytest.raises(ValidationError) as e:
        request(tools=[{"type": "function"}])
    assert [err["loc"] for err in e.value.errors()] == [("tools", 0, "function", "name")]


def test_chats_refusal_of_the_translation_is_a_bug(monkeypatch):
    validate = OpenAIChatRequest.model_validate
    monkeypatch.setattr(OpenAIChatRequest, "model_validate", lambda chat: validate({}))
    with pytest.raises(RuntimeError, match="a bug in tokin"):
        request()
