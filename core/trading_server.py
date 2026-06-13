"""HTTP order surface — a thin shell over the ONE guarded ``Broker`` (issue #9).

Ports moray's ``src/trading_server/`` shape: an HTTP endpoint that accepts order
requests and writes them. The crucial property is that it has NO order path of
its own — ``handle_order`` delegates to ``Broker.submit_order``, which routes
every order through the death-spiral guard. There is no un-guarded sell helper
on the server either.

Stdlib-only (``http.server``) so it imports cleanly in the toolchain-light test
suite. ``handle_order(payload)`` is pure and directly unit-testable without a
socket; the HTTP layer is a trivial wrapper over it.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional, Tuple

from core.broker import Broker, OrderRejectedError

logger = logging.getLogger("keel.trading_server")


class OrderRequestError(ValueError):
    """Raised when an incoming order payload is malformed."""


@dataclass
class TradingServer:
    """Wraps the one ``Broker``. ``handle_order`` is the single write path."""

    broker: Broker

    def handle_order(self, payload: dict) -> dict:
        """Validate a payload and route it through the guarded broker.

        Returns a JSON-serializable result dict. Raises ``OrderRequestError``
        on a malformed payload; a death-spiral refusal from the broker
        propagates as RuntimeError (the caller must not swallow it).
        """
        if not isinstance(payload, dict):
            raise OrderRequestError("payload must be a JSON object")
        try:
            symbol = str(payload["symbol"])
            side = str(payload["side"])
            qty = float(payload["qty"])
            price = float(payload["price"])
        except (KeyError, TypeError, ValueError) as exc:
            raise OrderRequestError(f"missing/invalid order field: {exc}") from exc

        order = self.broker.submit_order(symbol, side, qty, price)
        return {
            "status": "accepted",
            "symbol": order.symbol,
            "side": order.side,
            "qty": order.qty,
            "price": order.price,
        }


def make_handler(server: TradingServer):
    """Build a request handler bound to ``server`` (its broker)."""

    class _Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: dict) -> None:
            raw = json.dumps(body).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_POST(self):  # noqa: N802 - http.server API
            if self.path != "/order":
                self._send(404, {"error": "not found"})
                return
            length = int(self.headers.get("Content-Length", 0) or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw or b"{}")
                result = server.handle_order(payload)
            except (OrderRequestError, OrderRejectedError) as exc:
                self._send(400, {"error": str(exc)})
                return
            except RuntimeError as exc:
                # Death-spiral refusal (or other guard) — surface as 409, and
                # also let the operator see it loudly in the log.
                logger.error("[trading_server] order refused: %s", exc)
                self._send(409, {"error": str(exc)})
                return
            self._send(200, result)

        def log_message(self, *args):  # silence default stderr spam in tests
            pass

    return _Handler


def serve(broker: Optional[Broker] = None, host: str = "127.0.0.1",
          port: int = 8787) -> Tuple[HTTPServer, TradingServer]:
    """Start an HTTP order surface over a paper-default broker (demo helper)."""
    server = TradingServer(broker or Broker())
    httpd = HTTPServer((host, port), make_handler(server))
    logger.info("[trading_server] listening on %s:%d (paper=%s)",
                host, port, server.broker.paper)
    return httpd, server


def main():  # pragma: no cover - tiny demo harness, not a live entry point
    logging.basicConfig(level=logging.INFO)
    httpd, _ = serve()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        httpd.shutdown()


if __name__ == "__main__":
    main()


__all__ = ["TradingServer", "OrderRequestError", "make_handler", "serve"]
