import json
from http import HTTPStatus

import pytest
from conftest import said
from openai.types.responses.response_create_params import ResponseCreateParamsStreaming
from pydantic import ValidationError

from tokin.gateway import ChatResponse, ContextLengthExceededError
from tokin.harness_apis import OpenAIResponsesRequest
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
TOOL = {"type": "function", "name": "f", "description": "d", "parameters": {"type": "object"}, "strict": False}
CALL = {"id": "c1", "type": "function", "function": {"name": "f", "arguments": "{}"}}


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


def test_chat_request_params_leave_out_what_the_harness_did_not_set():
    assert request(temperature=0.7).chat_request().params == {"temperature": 0.7}


def test_max_output_tokens_is_max_tokens():
    assert request(max_output_tokens=5).chat_request().params == {"max_tokens": 5}


def test_parallel_tool_calls_without_tools_is_dropped():
    assert request(tools=[], parallel_tool_calls=False).chat_request().tools == []


def test_tools_become_chat_functions_in_the_order_sent():
    (tool,) = request(tools=[TOOL]).chat_request().tools
    assert list(tool) == ["type", "function"] and list(tool["function"]) == list(TOOL)[1:]


def test_leading_system_and_developer_messages_become_one_system_message():
    system = {"type": "message", "role": "system", "content": "a"}
    developer = {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "b"}]}
    assert request(input=[system, developer, USER]).chat_request().messages[:2] == [
        {"role": "system", "content": "a\n\nb"},
        {"role": "user", "content": "hi"},
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


def test_a_tool_other_than_a_function_is_refused():
    with pytest.raises(ValidationError, match="tools.0.type"):
        request(tools=[{"type": "web_search"}])


def test_an_item_of_another_type_is_refused():
    with pytest.raises(ValidationError, match="custom_tool_call"):
        request(input=[{"type": "custom_tool_call", "call_id": "c1", "name": "f", "input": ""}])


def test_parallel_tool_calls_false_with_tools_is_refused():
    with pytest.raises(ValidationError, match="parallel_tool_calls"):
        request(tools=[TOOL], parallel_tool_calls=False)


def test_chats_refusal_of_max_output_tokens_names_it():
    with pytest.raises(ValidationError, match="max_output_tokens"):
        request(max_output_tokens=0)


def test_chats_refusal_of_a_tool_is_located_in_it():
    with pytest.raises(ValidationError) as e:
        request(tools=[{"type": "function"}])
    assert e.value.errors()[0]["loc"] == ("tools", 0, "name")
