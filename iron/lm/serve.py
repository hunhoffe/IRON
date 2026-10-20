# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``serve``, a loaded model behind the OpenAI-compatible HTTP API that
Lemonade's backends speak: ``/v1/chat/completions`` for a causal model,
``/v1/embeddings`` for one that embeds, ``/v1/rerank`` for one that scores,
and ``/v1/models`` for each. One request runs on the device at a time; the
others wait. It takes nothing beyond the standard library.

    iron-serve qwen3 Qwen/Qwen3-0.6B --port 8001
"""

import argparse
import base64
import importlib
import json
import logging
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .causal import CausalLM
from .cli import parse_args
from .generation import Sampler

logger = logging.getLogger(__name__)

#: What a request may set of its ``Sampler``, by the request's field names.
SAMPLING = ("temperature", "top_k", "seed")


def chat(model, name: str, body: dict):
    """A chat completion of ``body["messages"]``, or with ``body["stream"]``
    its chunks, each drawn as the model draws its text. ``max_tokens``
    defaults to the rest of the context. ``usage`` carries Lemonade's
    ``prefill_duration_ttft`` and ``decoding_speed_tps`` beside the counts.
    """
    messages = body["messages"]
    prompt = len(model.tokenizer.chat(messages))
    most = model.config.max_seq_len - prompt
    max_tokens = body.get("max_completion_tokens", body.get("max_tokens", most))
    if not 0 < max_tokens <= most:
        raise ValueError(
            f"{prompt} tokens and {max_tokens} more do not fit {most + prompt}"
        )
    sampler = Sampler(**{k: body[k] for k in SAMPLING if k in body})
    pieces = model.stream(messages, max_tokens, sampler)
    reply = dict(
        id=f"chatcmpl-{uuid.uuid4().hex}", created=int(time.time()), model=name
    )

    def end():
        stats = model.stats
        usage = dict(
            prompt_tokens=prompt,
            completion_tokens=stats.tokens,
            total_tokens=prompt + stats.tokens,
            prefill_duration_ttft=stats.ttft,
        )
        if stats.tokens > 1:
            usage["decoding_speed_tps"] = stats.tokens_per_s
        return ("length" if stats.tokens == max_tokens else "stop"), usage

    if not body.get("stream"):
        text = "".join(pieces)
        finish, usage = end()
        message = dict(role="assistant", content=text)
        choice = dict(index=0, message=message, finish_reason=finish)
        return reply | dict(object="chat.completion", choices=[choice], usage=usage)

    # Draw the first piece now, so a request the model refuses is an error
    # status rather than a broken stream.
    first = next(pieces, None)

    def chunks():
        chunk = reply | dict(object="chat.completion.chunk")

        def event(delta, finish=None):
            choice = dict(index=0, delta=delta, finish_reason=finish)
            return chunk | dict(choices=[choice])

        yield event(dict(role="assistant", content=first or ""))
        for piece in pieces:
            yield event(dict(content=piece))
        finish, usage = end()
        yield event({}, finish) | dict(usage=usage)

    return chunks()


def embeddings(model, name: str, body: dict) -> dict:
    """The embedding of ``body["input"]``, a text or a list of them, as
    floats or, with ``encoding_format`` ``"base64"``, as the base64 of their
    little-endian float32 bytes.
    """
    texts = body["input"]
    texts = [texts] if isinstance(texts, str) else texts
    if not all(isinstance(t, str) for t in texts):
        raise TypeError("input: a text or a list of texts")
    base64_ = body.get("encoding_format", "float") == "base64"
    data = []
    for i, text in enumerate(texts):
        y = model.embed(text)
        vector = (
            base64.b64encode(y.astype("<f4").tobytes()).decode()
            if base64_
            else y.tolist()
        )
        data.append(dict(object="embedding", index=i, embedding=vector))
    return dict(object="list", data=data, model=name)


def rerank(model, name: str, body: dict) -> dict:
    """``model.score`` of each of ``body["documents"]`` against
    ``body["query"]``, in the documents' order.
    """
    query = body["query"]
    results = [
        dict(index=i, relevance_score=model.score(query, document))
        for i, document in enumerate(body["documents"])
    ]
    return dict(object="list", results=results, model=name)


def serve(model, name: str, host: str = "127.0.0.1", port: int = 8000):
    """A ``ThreadingHTTPServer`` on ``host`` and ``port`` (0 picks one),
    answering as ``name`` for ``model``, a loaded ``Model``: chat for a
    causal model, embeddings where it has ``embed``, reranking where it has
    ``score``. Start it with ``serve_forever()``.
    """
    routes = {}
    if isinstance(model, CausalLM):
        routes["/v1/chat/completions"] = chat
    if hasattr(model, "embed"):
        routes["/v1/embeddings"] = embeddings
    if hasattr(model, "score"):
        routes["/v1/rerank"] = rerank
    models = dict(object="list", data=[dict(id=name, object="model", owned_by="iron")])
    device = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_GET(self):
            if self.path == "/v1/models":
                self.send(200, models)
            else:
                self.error(404, f"no route {self.path}")

        def do_POST(self):
            route = routes.get(self.path)
            body = self.rfile.read(int(self.headers.get("Content-Length", 0)))
            if route is None:
                return self.error(404, f"{name} serves no {self.path}")
            with device:
                try:
                    request = json.loads(body)
                    # A reply names the model as its request did (Lemonade's name).
                    reply = route(model, request.get("model", name), request)
                except (
                    AttributeError,
                    KeyError,
                    TypeError,
                    ValueError,
                    IndexError,
                ) as e:
                    return self.error(400, f"{type(e).__name__}: {e}")
                except Exception as e:
                    logger.exception("%s failed", self.path)
                    return self.error(500, f"{type(e).__name__}: {e}")
                if isinstance(reply, dict):
                    return self.send(200, reply)
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for event in reply:
                    self.chunk(f"data: {json.dumps(event)}\n\n")
                self.chunk("data: [DONE]\n\n")
                self.chunk("")

        def chunk(self, text: str):
            """One chunk of a chunked body; the empty one ends it."""
            data = text.encode()
            self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
            self.wfile.flush()

        def send(self, status: int, reply: dict):
            data = json.dumps(reply).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def error(self, status: int, message: str):
            self.send(status, dict(error=dict(message=message, code=status)))

        def log_message(self, format, *args):
            logger.info(format, *args)

    return ThreadingHTTPServer((host, port), Handler)


def main(args=None):
    """``iron-serve``: an application's model, loaded, served until stopped."""
    parser = argparse.ArgumentParser(
        description="An IRON model behind an OpenAI-compatible HTTP API"
    )
    parser.add_argument(
        "application",
        help="a package of iron/applications, such as qwen3: its first "
        "export is the model",
    )
    parser.add_argument(
        "path",
        help="a directory holding config.json, the checkpoint and the "
        "tokenizer, or a Hugging Face repository's id; the name served",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        help="the longest context served (default: the most this host fits)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parse_args(parser, args)
    package = importlib.import_module(f"iron.applications.{args.application}")
    model_class = getattr(package, package.__all__[0])
    model = model_class.load(args.path, max_seq_len=args.max_seq_len)
    server = serve(model, args.path, args.host, args.port)
    print(f"Serving {model!r} at http://{args.host}:{server.server_port}/v1")
    server.serve_forever()


if __name__ == "__main__":
    main()
