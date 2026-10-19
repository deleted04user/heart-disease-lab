"""Wait for the container's native Streamlit process-health endpoint."""

import argparse
import time
from urllib.error import URLError
from urllib.parse import urlparse
from urllib.request import urlopen


def wait_for_health(url: str, timeout: float = 60) -> None:
    parsed = urlparse(url)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost"}:
        raise ValueError("Container verification accepts a local HTTP endpoint only.")
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            with urlopen(url.rstrip("/") + "/_stcore/health", timeout=3) as response:
                if response.status == 200 and response.read().strip() == b"ok":
                    return
        except (URLError, TimeoutError, OSError):
            pass
        time.sleep(1)
    raise RuntimeError("Container did not become healthy within its startup deadline.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8502")
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    wait_for_health(args.url, args.timeout)
    print("Container process health: passed. Run verify_cloud.py for model/UI readiness.")
