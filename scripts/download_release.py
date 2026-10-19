"""Download only the committed, checksum-pinned trusted public release archive."""

import json
from urllib.request import Request, urlopen

from _common import ROOT
from heart_disease.bundle import MAX_BUNDLE_BYTES, checksum

if __name__ == "__main__":
    config = json.loads((ROOT / "deploy/release.json").read_text(encoding="utf-8"))
    output = ROOT / "dist" / config["asset"]
    if output.exists():
        if checksum(output.read_bytes()) != config["sha256"]:
            raise ValueError(
                "An existing archive has a different checksum; it was not overwritten."
            )
    else:
        request = Request(
            config["url"], headers={"User-Agent": "HeartDiseaseLab/1.1 verified release client"}
        )
        with urlopen(request, timeout=45) as response:
            payload = response.read(MAX_BUNDLE_BYTES + 1)
        if len(payload) > MAX_BUNDLE_BYTES or checksum(payload) != config["sha256"]:
            raise ValueError("Downloaded release differs from its pinned checksum.")
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("xb") as handle:
            handle.write(payload)
    print(f"Verified release archive: {output.name} ({output.stat().st_size} bytes)")
