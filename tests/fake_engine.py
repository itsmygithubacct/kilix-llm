"""Stand-ins for llama-server and plebian-model-sizer in the judge tests.

`fake_engine.py server ARGS...` behaves like llama-server on the unix socket
named by --host, with one token per character and a fixed chat template.
`fake_engine.py sizer ARGS...` prints FAKE_SIZER_JSON and exits FAKE_SIZER_RC.
Both append what they were given to FAKE_LOG (JSON lines), so a test can
check the exact argv and request bodies. No model, no network.

FAKE_ANSWER: "0", "1", "x" (outside the contract), "crash" (exit at startup) or
"die" (exit without answering, mid-completion).
FAKE_LOGPROBS: JSON list of [token, logprob] candidates for the answer.

For the stack tests the server also answers `/v1/chat/completions` (the draft
for seed 1000+i is item i of the JSON list FAKE_DRAFTS, or "draft i") and
`/v1/systemone` (each question's rating is the FAKE_RATINGS value whose key
its instructions contain, or 0.5; FAKE_SYSTEMONE=404 removes the endpoint).
FAKE_P0 is a JSON object: a `/completion` prompt containing one of its keys is
answered with that probability of "0". FAKE_MAX_PARALLEL makes a server asked
for more slots exit at startup.
"""

import http.server
import json
import math
import os
import socketserver
import sys

TEMPLATE = "<|s|>{system}<|u|>{user}<|a|>"


def log(kind, value):
    path = os.environ.get("FAKE_LOG")
    if path:
        with open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, "value": value}) + "\n")


def tokens(text):
    return [ord(char) for char in text]


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *_args):
        pass

    def address_string(self):
        return "unix"

    def reply(self, value, status=200):
        data = json.dumps(value).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        if self.path == "/health":
            self.reply({"status": "ok"})
        else:
            self.reply({"error": "not found"}, 404)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        log(self.path, body)
        if self.path == "/apply-template":
            messages = {m["role"]: m["content"] for m in body["messages"]}
            self.reply({"prompt": TEMPLATE.format(system=messages["system"], user=messages["user"])})
        elif self.path == "/tokenize":
            self.reply({"tokens": tokens(body["content"])})
        elif self.path == "/detokenize":
            self.reply({"content": "".join(chr(t) for t in body["tokens"])})
        elif self.path == "/completion":
            answer = os.environ.get("FAKE_ANSWER", "0")
            if answer == "die":
                print("fake engine: killed during the prompt", file=sys.stderr, flush=True)
                os._exit(9)
            candidates = json.loads(os.environ.get("FAKE_LOGPROBS", '[["0", -0.5], ["1", -1.5]]'))
            if os.environ.get("FAKE_P0"):
                text = "".join(chr(t) for t in body["prompt"])
                p0 = next((p for key, p in json.loads(os.environ["FAKE_P0"]).items() if key in text), 0.5)
                candidates = [["0", math.log(p0)], ["1", math.log(1 - p0)]]
                answer = "0" if p0 >= 0.5 else "1"
            self.reply({
                "content": answer,
                "tokens_evaluated": len(body["prompt"]),
                "completion_probabilities": [{
                    "token": answer, "logprob": dict(candidates).get(answer, -0.1),
                    "top_logprobs": [{"token": t, "logprob": p} for t, p in candidates]}],
            })
        elif self.path == "/v1/chat/completions":
            slot = body["seed"] - 1000
            drafts = json.loads(os.environ.get("FAKE_DRAFTS", "[]"))
            content = drafts[slot] if slot < len(drafts) else f"draft {slot}"
            if content is None:
                self.reply({"error": "slot failed"}, 500)
                return
            self.reply({"id": "chatcmpl-fake", "created": 1, "model": "fake", "choices": [{
                "index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}],
                "usage": {"completion_tokens": len(content)}})
        elif self.path == "/v1/systemone" and os.environ.get("FAKE_SYSTEMONE") != "404":
            ratings = json.loads(os.environ.get("FAKE_RATINGS", "{}"))
            self.reply({"answers": {name: {"noul": next(
                (value for key, value in ratings.items() if key in question["instructions"]), 0.5)}
                for name, question in body["questions"].items()}})
        else:
            self.reply({"error": "not found"}, 404)


class Server(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True

    def get_request(self):
        request, _ = super().get_request()
        return request, ("unix", 0)


def server(argv):
    log("server-argv", argv)
    if os.environ.get("FAKE_ANSWER") == "crash":
        print("fake engine: failed to load model", flush=True)
        return 3
    limit = os.environ.get("FAKE_MAX_PARALLEL")
    if limit and int(argv[argv.index("--parallel") + 1]) > int(limit):
        print("fake engine: failed to create context", flush=True)
        return 1
    host = argv[argv.index("--host") + 1]
    with Server(host, Handler) as httpd:
        httpd.serve_forever()
    return 0


def sizer(argv):
    log("sizer-argv", argv)
    print(os.environ.get("FAKE_SIZER_JSON", "{}"))
    return int(os.environ.get("FAKE_SIZER_RC", "0"))


if __name__ == "__main__":
    kind, rest = sys.argv[1], sys.argv[2:]
    raise SystemExit(server(rest) if kind == "server" else sizer(rest))
