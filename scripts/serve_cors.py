#!/usr/bin/env python3
"""Simple static file server with permissive CORS headers."""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from functools import partial


class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    """Serve files and add wide-open CORS headers."""

    def end_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Range")
        super().end_headers()

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        # Quieter logging.
        return

    def do_OPTIONS(self) -> None:
        self.send_response(200, "ok")
        self.end_headers()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Serve static files with permissive CORS headers."
    )
    parser.add_argument("--port", type=int, default=9000, help="Port to listen on.")
    parser.add_argument(
        "--dir",
        type=str,
        default=".",
        help="Directory to serve (defaults to current working directory).",
    )
    args = parser.parse_args()

    os.chdir(args.dir)
    handler = partial(CORSRequestHandler, directory=args.dir)
    with socketserver.ThreadingTCPServer(("", args.port), handler) as httpd:
        print(f"Serving {os.path.abspath(args.dir)} on port {args.port} with CORS")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
