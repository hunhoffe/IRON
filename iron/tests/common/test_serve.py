# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``serve``'s API over HTTP: chat from a toy Llama repository, the host
standing in for the NPU, and embeddings and reranking from stand-ins whose
numbers their own applications' tests check.
"""

import base64
import json
import threading
import urllib.error
import urllib.request

import numpy as np
import pytest

from iron.lm import Sampler
from iron.lm.serve import serve
from iron.tests.common import test_llama_reference as reference
from iron.tests.support.llama_reference import OnHost

pytestmark = pytest.mark.usefixtures("npu2")  # MHA decode on any host

HI = [{"role": "user", "content": "The capital of France is"}]


def _start(model, name="toy"):
    """``model`` served on a free port, and the URL of its ``/v1``."""
    server = serve(model, name, port=0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_port}/v1"


def _post(url, body):
    """The status and the parsed or raw reply of POSTing ``body`` as JSON."""
    request = urllib.request.Request(url, json.dumps(body).encode(), method="POST")
    request.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(request) as reply:
            data = reply.read().decode()
            if reply.headers["Content-Type"] == "text/event-stream":
                return reply.status, data
            return reply.status, json.loads(data)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


@pytest.fixture(scope="module")
def llama(tmp_path_factory):
    directory = tmp_path_factory.mktemp("repository")
    reference._repository(directory, reference.TestLlama.weights())
    settings = {"chat_template": reference.CHATML}
    (directory / "tokenizer_config.json").write_text(json.dumps(settings))
    pytest.importorskip("jinja2")
    model = OnHost.load(directory)
    server, url = _start(model)
    yield model, url
    server.shutdown()


def test_a_chat_completion_is_what_generate_gives(llama):
    model, url = llama
    body = dict(model="Qwen3-0.6B-IRON", messages=HI, max_tokens=5, seed=3)
    status, reply = _post(f"{url}/chat/completions", body)
    assert reply["model"] == "Qwen3-0.6B-IRON"
    want = model.generate(HI, 5, Sampler(seed=3))
    assert status == 200 and reply["choices"][0]["message"]["content"] == want
    usage = reply["usage"]
    assert usage["prompt_tokens"] == len(model.tokenizer.chat(HI))
    assert (
        usage["completion_tokens"] == 5
        and reply["choices"][0]["finish_reason"] == "length"
    )


def test_a_streamed_chat_s_deltas_join_to_the_completion(llama):
    model, url = llama
    body = dict(messages=HI, max_tokens=5, seed=3, stream=True)
    status, data = _post(f"{url}/chat/completions", body)
    events = [line.removeprefix("data: ") for line in data.split("\n\n") if line]
    assert status == 200 and events[-1] == "[DONE]"
    chunks = [json.loads(e) for e in events[:-1]]
    text = "".join(c["choices"][0]["delta"].get("content", "") for c in chunks)
    assert text == model.generate(HI, 5, Sampler(seed=3))
    assert chunks[0]["choices"][0]["delta"]["role"] == "assistant"
    assert chunks[-1]["choices"][0]["finish_reason"] == "length"
    assert chunks[-1]["usage"]["completion_tokens"] == 5


@pytest.mark.parametrize(
    "body",
    [[HI], dict(max_tokens=5), dict(messages=HI, max_tokens=1 << 20)],
    ids=["not an object", "no messages", "past the context"],
)
def test_a_request_the_model_refuses_is_a_400(llama, body):
    _, url = llama
    for stream in (False, True):
        if isinstance(body, dict):
            body = body | dict(stream=stream)
        status, reply = _post(f"{url}/chat/completions", body)
        assert status == 400 and reply["error"]["message"]


def test_a_route_the_model_lacks_is_a_404(llama):
    _, url = llama
    status, _ = _post(f"{url}/embeddings", dict(input="x"))
    assert status == 404


def test_models_lists_the_name_served(llama):
    _, url = llama
    with urllib.request.urlopen(f"{url}/models") as reply:
        assert [m["id"] for m in json.load(reply)["data"]] == ["toy"]


class Embedder:
    def embed(self, text):
        return np.array([len(text), 1.5], np.float32)


class Reranker:
    def score(self, query, document):
        return float(len(set(query.split()) & set(document.split())))


def test_embeddings_come_as_floats_or_base64():
    server, url = _start(Embedder())
    _, floats = _post(f"{url}/embeddings", dict(input=["ab", "abc"]))
    _, packed = _post(f"{url}/embeddings", dict(input="ab", encoding_format="base64"))
    server.shutdown()
    assert [d["embedding"] for d in floats["data"]] == [[2, 1.5], [3, 1.5]]
    raw = base64.b64decode(packed["data"][0]["embedding"])
    assert np.frombuffer(raw, "<f4").tolist() == [2, 1.5]


def test_rerank_scores_each_document_in_order():
    server, url = _start(Reranker())
    body = dict(query="north sky", documents=["the sky", "the sea", "north sky"])
    status, reply = _post(f"{url}/rerank", body)
    server.shutdown()
    assert status == 200
    assert [(r["index"], r["relevance_score"]) for r in reply["results"]] == [
        (0, 1.0),
        (1, 0.0),
        (2, 2.0),
    ]
