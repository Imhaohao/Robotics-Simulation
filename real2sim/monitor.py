"""Serve the eval monitor. The page polls outputs/progress.json."""

from __future__ import annotations

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Watch trekion_real2sim_v1 while it runs")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--root", type=Path, default=Path("."))
    args = parser.parse_args()
    root = args.root.resolve()

    class Handler(SimpleHTTPRequestHandler):
        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(root), **kw)

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                self.path = "/monitor/index.html"
            super().do_GET()

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    print(f"monitor http://127.0.0.1:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
