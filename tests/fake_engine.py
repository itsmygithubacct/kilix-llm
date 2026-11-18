"""Stand-ins for llama-server and plebian-model-sizer in the judge tests.

`fake_engine.py server ARGS...` behaves like llama-server on the unix socket
named by --host, with one token per character and a fixed chat template.
`fake_engine.py sizer ARGS...` prints FAKE_SIZER_JSON and exits FAKE_SIZER_RC.
Both append what they were given to FAKE_LOG (JSON lines), so a test can
check the exact argv and request bodies. No model, no network.

FAKE_ANSWER: "0", "1", "x" (outside the contract) or "crash" (exit at startup).
FAKE_LOGPROBS: JSON list of [token, logprob] candidates for the answer.
"""

import http.server
import json
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
            candidates = json.loads(os.environ.get("FAKE_LOGPROBS", '[["0", -0.5], ["1", -1.5]]'))
            self.reply({
                "content": answer,
                "tokens_evaluated": len(body["prompt"]),
                "completion_probabilities": [{
                    "token": answer, "logprob": dict(candidates).get(answer, -0.1),
                    "top_logprobs": [{"token": t, "logprob": p} for t, p in candidates]}],
            })
        else:
            self.reply({"error": "not found"}, 404)


class Server(socketserver.UnixStreamServer):
    def get_request(self):
        request, _ = super().get_request()
        return request, ("unix", 0)


def server(argv):
    log("server-argv", argv)
    if os.environ.get("FAKE_ANSWER") == "crash":
        print("fake engine: failed to load model", flush=True)
        return 3
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
