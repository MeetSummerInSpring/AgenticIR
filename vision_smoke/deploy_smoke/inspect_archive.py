#!/usr/bin/env python3
"""Inspect the platform and payload of a Docker archive without Docker."""

import argparse
import json
import tarfile
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("archive", type=Path)
    args = parser.parse_args()
    with tarfile.open(args.archive, "r") as archive:
        manifest = json.load(archive.extractfile("manifest.json"))
        if len(manifest) != 1:
            raise RuntimeError("expected exactly one image in archive")
        record = manifest[0]
        config = json.load(archive.extractfile(record["Config"]))
        names = set(archive.getnames())
        if not all(layer in names for layer in record["Layers"]):
            raise RuntimeError("archive is missing one or more layers")
        if not any(layer.endswith("/layer.tar") and "app/smoke.py" in tarfile.open(fileobj=archive.extractfile(layer), mode="r:").getnames() for layer in record["Layers"]):
            raise RuntimeError("archive does not contain /app/smoke.py")
    print("image=%s" % ",".join(record.get("RepoTags", [])))
    print("os=%s" % config.get("os"))
    print("architecture=%s" % config.get("architecture"))
    print("working_dir=%s" % config.get("config", {}).get("WorkingDir"))
    print("cmd=%s" % json.dumps(config.get("config", {}).get("Cmd")))
    if config.get("os") != "linux" or config.get("architecture") != "arm64":
        raise RuntimeError("archive platform is not linux/arm64")
    print("status=archive validation passed")


if __name__ == "__main__":
    main()
