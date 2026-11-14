"""Build and persist the PaySim graph payload for the laptop API."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.api import routes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paysim", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--row-cap", type=int, default=250000)
    args = parser.parse_args()
    source = args.paysim.resolve()
    stat = source.stat()
    previous_path = routes.PAYSIM_PATH
    previous_laptop = os.environ.get("FRAUDGRAPH_LAPTOP_MODE")
    previous_live = os.environ.get("FRAUDGRAPH_GRAPH_LIVE_BUILD")
    try:
        routes.PAYSIM_PATH = source
        os.environ["FRAUDGRAPH_LAPTOP_MODE"] = "0"
        os.environ["FRAUDGRAPH_GRAPH_LIVE_BUILD"] = "1"
        response = routes.paysim_graph(
            limit=500,
            max_rows=args.row_cap,
            dataset_id=None,
            account_id=None,
            neighbor_limit=24,
            owner_id="legacy",
        )
    finally:
        routes.PAYSIM_PATH = previous_path
        if previous_laptop is None:
            os.environ.pop("FRAUDGRAPH_LAPTOP_MODE", None)
        else:
            os.environ["FRAUDGRAPH_LAPTOP_MODE"] = previous_laptop
        if previous_live is None:
            os.environ.pop("FRAUDGRAPH_GRAPH_LIVE_BUILD", None)
        else:
            os.environ["FRAUDGRAPH_GRAPH_LIVE_BUILD"] = previous_live
    payload = {
        "metadata": {
            "source_file_size": stat.st_size,
            "source_mtime": stat.st_mtime,
            "source_mtime_ns": stat.st_mtime_ns,
            "row_cap": args.row_cap,
            "sampled": response["sampled"],
            "sampling_ratio": response["sampling_ratio"],
            "built_at": datetime.now(timezone.utc).isoformat(),
            "source_size": stat.st_size,
        },
        "response": response,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
