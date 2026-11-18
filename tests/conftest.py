import base64
import io

import jinja2
import pytest
from PIL import Image

from tokin.chat_templates import ChatTemplate
from tokin.gateway import Gateway
from tokin.inference_engines import InferenceEngine
from tokin.rollout import FinishReason, Generation

# One private-use character per control token, so each is a single id.
START, END, PAD = "", "", ""
CHATML = (
    "{%- for m in messages %}{{ start ~ m.role ~ '\\n' }}"
    "{%- if m.role == 'system' and tools %}tools={{ tools | tojson }}{% endif %}"
    "{%- if m.content is string or m.content is none %}{{ m.content }}"
    "{%- else %}{% for p in m.content %}{{ p.text if p.type == 'text' else pad }}{% endfor %}{% endif %}"
    "{%- for c in m.tool_calls or [] %}<call>{{ c.function.arguments | tojson }}</call>{% endfor %}"
    "{%- if m.tool_call_id %}<id>{{ m.tool_call_id }}</id>{% endif %}"
    "{{ end ~ '\\n' }}{%- endfor %}"
    "{%- if add_generation_prompt %}{{ start ~ 'assistant\\n' }}"
    "{%- if enable_thinking is defined and not enable_thinking %}<think></think>{% endif %}{% endif %}"
)


class FakeTokenizer:
    def __init__(self, template: str = CHATML) -> None:
        self.chat_template = template
        self._env = jinja2.Environment(extensions=["jinja2.ext.loopcontrols"])

    def apply_chat_template(self, messages, *, tools=None, add_generation_prompt, tokenize, **kwargs):
        def raise_exception(message):
            raise jinja2.TemplateError(message)

        return self._env.from_string(self.chat_template).render(
            messages=messages,
            tools=tools,
            add_generation_prompt=add_generation_prompt,
            raise_exception=raise_exception,
            start=START,
            end=END,
            pad=PAD,
            **kwargs,
        )

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(map(chr, ids))


class Pixels:
    """An image processor that counts a token per pixel."""

    merge_size = 1

    def get_number_of_image_patches(self, height, width, kwargs):
        return height * width


def image(width, height):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height)).save(buffer, "PNG")
    url = f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"
    return {"type": "image_url", "image_url": {"url": url}}


class ChatML(ChatTemplate):
    name = "chatml"
    turn_end = f"{END}\n"
    reasoning_start = "<think>"
    reasoning_end = "</think>"
    tool_call_parser = "hermes"
    stop = (END,)
    kwargs = ("enable_thinking", "add_vision_id")
    image_pad = PAD
    _image_processor = Pixels()


@pytest.fixture
def chatml():
    return ChatML(FakeTokenizer())


class Engine(InferenceEngine):
    def __init__(self, *replies, context_length=1000, routing=None):
        self.replies, self.calls, self.length, self.row = list(replies), [], context_length, routing
        self.images = []

    async def ready(self):
        pass

    async def context_length(self):
        return self.length

    async def routing(self):
        return self.row or await super().routing()

    async def generate(self, token_ids, params, images=()):
        self.calls.append((token_ids, params))
        self.images.append(list(images))
        return self.replies.pop(0)


def said(text, finish=FinishReason.STOP):
    return Generation([ord(c) for c in text] + [ord(END)] * (finish is FinishReason.STOP), finish)


@pytest.fixture
def gateway(chatml):
    def make(*replies, context_length=1000, routing=None):
        return Gateway(chatml, Engine(*replies, context_length=context_length, routing=routing))

    return make
