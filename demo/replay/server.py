from __future__ import annotations

from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


WEB_ROOT = Path(__file__).resolve().parent / "web"


class ReplayRequestHandler(SimpleHTTPRequestHandler):
    def end_headers(self) -> None:
        if self.path.endswith("manifest.json"):
            self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def log_message(self, format: str, *args: object) -> None:
        print(f"[replay] {self.address_string()} - {format % args}")


def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    handler = partial(ReplayRequestHandler, directory=str(WEB_ROOT))
    server = ThreadingHTTPServer((host, port), handler)
    print(f"AgenticIR 回放演示：http://{host}:{port}")
    print("按 Ctrl+C 停止服务")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n服务已停止")
    finally:
        server.server_close()
