#!/usr/bin/env python
"""Create and revoke FraudGraph API keys."""

from __future__ import annotations

import argparse

from src.auth import create_api_key_record, revoke_api_key


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    create = subparsers.add_parser("create")
    create.add_argument("--owner-id", required=True)
    revoke = subparsers.add_parser("revoke")
    revoke.add_argument("--key-id", required=True)
    args = parser.parse_args()
    if args.command == "create":
        key_id, raw_key = create_api_key_record(args.owner_id)
        print(f"key_id: {key_id}")
        print(f"api_key: {raw_key}")
        return 0
    if not revoke_api_key(args.key_id):
        parser.error("key not found or already revoked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
