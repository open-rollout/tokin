<h1 align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/header-dark.svg">
    <img src="assets/header-light.svg" alt="tokin. Make any agent harness token-native." width="480">
  </picture>
</h1>

<p align="center">
  <a href="https://github.com/open-rollout/tokin/actions/workflows/ci.yml"><img src="https://github.com/open-rollout/tokin/actions/workflows/ci.yml/badge.svg?branch=main" alt="CI"></a>
  <a href="https://pypi.org/project/tokin/"><img src="https://img.shields.io/pypi/v/tokin" alt="PyPI"></a>
  <a href="LICENSE"><img src="https://img.shields.io/github/license/open-rollout/tokin" alt="License"></a>
  <a href="https://www.yuanhe.wiki/posts/technical/tokin/"><img src="https://img.shields.io/badge/blog-yuanhe.wiki-blue" alt="Blog"></a>
</p>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="assets/gateway-dark.svg">
    <img src="assets/gateway-light.svg" alt="tokin as a gateway. Harnesses, any harness, each call their own session URL with chat completions in text. Inside tokin each turn is attached to its place in its session's tree; rendered, keeping the ids the model sampled for rollouts, rollout and s, where re-encoding the text would split it as roll and outs; generated on the inference engine from input tokens to output tokens; and parsed back into an assistant message, {&quot;role&quot;: &quot;assistant&quot;, …}, with reasoning and tool calls for its harness. The turn's ids grow the session's tree at that place, each leaf a rollout with token_ids, loss_mask and logprobs, which a trainer collects to update the weights the engine serves." width="100%">
  </picture>
</p>

**tokin** makes any agent harness token-native. Train an agent in the harness it ships with, on exactly the tokens
it sampled. Point the harness at tokin instead of the inference engine and nothing else changes: it keeps speaking
text over chat completions, while the engine only ever sees token ids. tokin keeps each session as a tree of turns, so
every rollout a trainer collects is token-in, token-out at every prefix.

## Quick start

Serve a model with sglang, and put tokin in front of it, each in its own terminal:

```bash
pip install tokin
python -m sglang.launch_server --model-path Qwen/Qwen3.5-0.8B --port 30000
tokin serve --model-path Qwen/Qwen3.5-0.8B --engine-url http://localhost:30000
```

Open a session for each sample. It prints the base URL the harness takes:

```bash
URL=$(tokin open --params '{"return_logprobs": true}')
```

For [pi](https://pi.dev), add that URL as a provider in `~/.pi/agent/models.json`, then run
pi as usual:

```json
{
  "providers": {
    "tokin": {
      "baseUrl": "<URL>",
      "api": "openai-completions",
      "apiKey": "tokin",
      "models": [{ "id": "Qwen/Qwen3.5-0.8B" }]
    }
  }
}
```

```bash
pi --provider tokin --model Qwen/Qwen3.5-0.8B "Write a haiku about tokens to haiku.txt."
```

Collect the session, one JSON line per rollout with its `token_ids`, `loss_mask` and
`logprobs`:

```bash
tokin collect "$URL" -o rollouts.jsonl
```

## Examples

- [`examples/harbor_eval`](examples/harbor_eval): evaluate mini-swe-agent, opencode or pi through tokin on [Harbor](https://github.com/harbor-framework/harbor).
- [`examples/pi_mimo_rl`](examples/pi_mimo_rl): train pi with slime on MiMo-V2.6-RL-oss.
