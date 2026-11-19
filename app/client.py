"""OLMo-2 Inference Server Network Client.

A reusable Python client for the /generate, /health, /validate, and /metrics
endpoints. This is what you run from a laptop or any other machine on your LAN
to test the full client → network → API → GPU → response chain.

Usage:
    # Quick one-shot generation from the command line
    python -m app.client --server http://192.168.1.100:8000 --prompt "Explain attention mechanisms"

    # Or import and use in code
    from app.client import OLMoServerClient
    client = OLMoServerClient("http://192.168.1.100:8000")
    print(client.generate("Hello world"))
"""

import argparse
import json
import time
import urllib.request
import urllib.error
from typing import Any, Dict, List, Optional


class OLMoServerClient:
    """HTTP client for the OLMo-2 inference server API.

    Uses Python's built-in urllib so there are no extra dependencies.
    Works from any machine that can reach the server IP over the network.
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: int = 120):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def _post(self, endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach server at {url}.\n"
                f"Reason: {e.reason}\n"
                "Check that the server is running and the IP/port is correct."
            ) from e

    def _get(self, endpoint: str) -> Dict[str, Any]:
        url = f"{self.base_url}{endpoint}"
        req = urllib.request.Request(url, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8")
            raise RuntimeError(f"HTTP {e.code} from {url}: {body}") from e
        except urllib.error.URLError as e:
            raise ConnectionError(
                f"Cannot reach server at {url}.\nReason: {e.reason}"
            ) from e

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def health(self) -> Dict[str, Any]:
        """Check server health, model status, and uptime."""
        return self._get("/health")

    def metrics(self) -> Dict[str, Any]:
        """Get current GPU memory, request count, and average latency."""
        return self._get("/metrics")

    def validate(self, prompt: str, max_new_tokens: int = 128) -> Dict[str, Any]:
        """Pre-flight check: verify a prompt fits within the context window."""
        return self._post("/validate", {
            "prompt": prompt,
            "requested_max_new_tokens": max_new_tokens,
        })

    def generate(
        self,
        prompt: Optional[str] = None,
        messages: Optional[List[Dict[str, str]]] = None,
        max_new_tokens: int = 128,
        temperature: float = 0.7,
        top_p: float = 0.9,
        disable_eos: bool = False,
    ) -> Dict[str, Any]:
        """Send a generation request and return the full response dict."""
        payload: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "disable_eos": disable_eos,
        }
        if messages:
            payload["messages"] = messages
            payload["prompt"] = ""
        else:
            payload["prompt"] = prompt or ""

        return self._post("/generate", payload)

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Convenience wrapper that returns just the generated text string."""
        result = self.generate(prompt=prompt, **kwargs)
        return result["generated_text"]


# ---------------------------------------------------------------------------
# CLI entrypoint — run from another machine to test the full network chain
# ---------------------------------------------------------------------------
def _print_divider(label: str = "") -> None:
    line = "=" * 60
    if label:
        print(f"\n{line}")
        print(f"  {label}")
        print(line)
    else:
        print(line)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="OLMo-2 Server Network Client — tests the full client → network → API → GPU chain."
    )
    parser.add_argument(
        "--server",
        type=str,
        default="http://localhost:8000",
        help="Server base URL (e.g. http://192.168.1.100:8000)",
    )
    parser.add_argument(
        "--prompt", type=str, default="Explain how neural networks learn from data.",
        help="Text prompt to generate from."
    )
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument(
        "--mode",
        choices=["generate", "health", "metrics", "validate", "all"],
        default="all",
        help="Which endpoint(s) to call.",
    )
    args = parser.parse_args()

    client = OLMoServerClient(base_url=args.server)

    print(f"\nConnecting to server: {args.server}")

    if args.mode in ("health", "all"):
        _print_divider("GET /health")
        try:
            h = client.health()
            print(f"  Status      : {h.get('status')}")
            print(f"  Uptime      : {h.get('uptime_seconds', 0):.1f} seconds")
            print(f"  Model       : {h.get('model_path')}")
            print(f"  Device      : {h.get('device')}")
            print(f"  dtype       : {h.get('torch_dtype')}")
            print(f"  Requests    : {h.get('request_count', 0)}")
            print(f"  Avg Latency : {h.get('avg_latency_seconds')} seconds")
        except (ConnectionError, RuntimeError) as e:
            print(f"  ERROR: {e}")

    if args.mode in ("metrics", "all"):
        _print_divider("GET /metrics")
        try:
            m = client.metrics()
            hw = m.get("hardware", {})
            print(f"  GPU         : {hw.get('cuda_device_name', 'N/A')}")
            print(f"  VRAM Used   : {hw.get('cuda_allocated_mb', 'N/A')} MB")
            print(f"  VRAM Rsrvd  : {hw.get('cuda_reserved_mb', 'N/A')} MB")
            print(f"  Requests    : {m.get('request_count', 0)}")
            print(f"  Avg Latency : {m.get('avg_latency_seconds')} seconds")
        except (ConnectionError, RuntimeError) as e:
            print(f"  ERROR: {e}")

    if args.mode in ("validate", "all"):
        _print_divider("POST /validate")
        try:
            v = client.validate(args.prompt, max_new_tokens=args.max_new_tokens)
            print(f"  Prompt Tokens           : {v.get('prompt_token_count')}")
            print(f"  Requested max_new_tokens: {v.get('requested_max_new_tokens')}")
            print(f"  Context Limit           : {v.get('context_limit')}")
            print(f"  Fits                    : {v.get('fits')}")
            print(f"  Recommended max_tokens  : {v.get('recommended_max_new_tokens')}")
            if v.get("warning"):
                print(f"  WARNING: {v.get('warning')}")
        except (ConnectionError, RuntimeError) as e:
            print(f"  ERROR: {e}")

    if args.mode in ("generate", "all"):
        _print_divider("POST /generate")
        try:
            t0 = time.time()
            result = client.generate(
                prompt=args.prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            client_latency = time.time() - t0

            print(f"  Generated Text      : {result.get('generated_text', '')}")
            print()
            print(f"  Prompt Tokens       : {result.get('prompt_token_count')}")
            print(f"  Response Tokens     : {result.get('response_token_count')}")
            print(f"  Finish Reason       : {result.get('finish_reason')}")
            print(f"  Server Latency      : {result.get('total_latency_seconds', 0.0):.3f} seconds")
            print(f"  Client E2E Latency  : {client_latency:.3f} seconds  (includes network round-trip)")
            network_overhead = client_latency - (result.get("total_latency_seconds") or 0.0)
            print(f"  Network Overhead    : {network_overhead:.3f} seconds")
            print(f"  Throughput          : {result.get('tokens_per_sec', 0.0):.2f} tokens/second")
        except (ConnectionError, RuntimeError) as e:
            print(f"  ERROR: {e}")

    _print_divider()
    print()
