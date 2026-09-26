#!/usr/bin/env python3
"""Create a linux/arm64 Docker archive without a local Docker daemon."""

import argparse
import datetime as dt
import gzip
import hashlib
import io
import json
import os
import ssl
import tarfile
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path


REGISTRY = "https://registry-1.docker.io"
REPOSITORY = "library/python"
BASE_TAG = "3.10-slim"
IMAGE_TAG = "vision-runtime:smoke"
INDEX_TYPES = ", ".join(
    (
        "application/vnd.oci.image.index.v1+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
    )
)
MANIFEST_TYPES = ", ".join(
    (
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.docker.distribution.manifest.v2+json",
    )
)


def request_json(url, token=None, accept=None):
    headers = {"User-Agent": "vision-runtime-builder/1.0"}
    if token:
        headers["Authorization"] = "Bearer " + token
    if accept:
        headers["Accept"] = accept
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=120) as response:
        return json.load(response)


def get_token():
    query = urllib.parse.urlencode(
        {"service": "registry.docker.io", "scope": "repository:%s:pull" % REPOSITORY}
    )
    response = request_json("https://auth.docker.io/token?" + query)
    return response["token"]


def get_manifest(reference, token, accept):
    url = "%s/v2/%s/manifests/%s" % (REGISTRY, REPOSITORY, reference)
    return request_json(url, token=token, accept=accept)


def select_arm64(index):
    for descriptor in index.get("manifests", []):
        platform = descriptor.get("platform", {})
        if platform.get("os") == "linux" and platform.get("architecture") == "arm64":
            return descriptor["digest"]
    raise RuntimeError("the base image index has no linux/arm64 manifest")


def download_blob(digest, token, destination):
    url = "%s/v2/%s/blobs/%s" % (REGISTRY, REPOSITORY, digest)
    headers = {"Authorization": "Bearer " + token, "User-Agent": "vision-runtime-builder/1.0"}
    request = urllib.request.Request(url, headers=headers)
    hasher = hashlib.sha256()
    with urllib.request.urlopen(request, timeout=300) as response, destination.open("wb") as output:
        while True:
            block = response.read(1024 * 1024)
            if not block:
                break
            hasher.update(block)
            output.write(block)
    expected = digest.partition(":")[2]
    if hasher.hexdigest() != expected:
        raise RuntimeError("downloaded blob digest mismatch: %s" % digest)


def json_bytes(value):
    return json.dumps(value, separators=(",", ":"), sort_keys=True).encode("utf-8")


def make_app_layer(script_path):
    output = io.BytesIO()
    timestamp = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    with tarfile.open(fileobj=output, mode="w", format=tarfile.PAX_FORMAT) as archive:
        directory = tarfile.TarInfo("app")
        directory.type = tarfile.DIRTYPE
        directory.mode = 0o755
        directory.uid = directory.gid = 0
        directory.uname = directory.gname = "root"
        directory.mtime = timestamp
        archive.addfile(directory)

        script = script_path.read_bytes()
        entry = tarfile.TarInfo("app/smoke.py")
        entry.size = len(script)
        entry.mode = 0o755
        entry.uid = entry.gid = 0
        entry.uname = entry.gname = "root"
        entry.mtime = timestamp
        archive.addfile(entry, io.BytesIO(script))
    return output.getvalue()


def decompress_layer(blob_path, media_type):
    data = blob_path.read_bytes()
    if media_type.endswith("+gzip") or data.startswith(b"\x1f\x8b"):
        return gzip.decompress(data)
    if media_type.endswith(".tar"):
        return data
    raise RuntimeError("unsupported base layer compression: %s" % media_type)


def chain_ids(diff_ids):
    result = []
    parent = None
    for diff_id in diff_ids:
        if parent is None:
            current = diff_id.split(":", 1)[1]
        else:
            material = ("sha256:%s %s" % (parent, diff_id)).encode("ascii")
            current = hashlib.sha256(material).hexdigest()
        result.append(current)
        parent = current
    return result


def add_bytes(archive, name, data, mode=0o644):
    entry = tarfile.TarInfo(name)
    entry.size = len(data)
    entry.mode = mode
    entry.uid = entry.gid = 0
    entry.uname = entry.gname = "root"
    entry.mtime = int(os.environ.get("SOURCE_DATE_EPOCH", "0"))
    archive.addfile(entry, io.BytesIO(data))


def build(output_path, script_path):
    token = get_token()
    index = get_manifest(BASE_TAG, token, INDEX_TYPES)
    if "manifests" in index:
        manifest_digest = select_arm64(index)
        manifest = get_manifest(manifest_digest, token, MANIFEST_TYPES)
    else:
        manifest = index
    config_descriptor = manifest["config"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="vision-runtime-build-") as temporary:
        temporary_path = Path(temporary)
        config_blob = temporary_path / "config.json"
        download_blob(config_descriptor["digest"], token, config_blob)
        config = json.loads(config_blob.read_text(encoding="utf-8"))
        if config.get("architecture") != "arm64" or config.get("os") != "linux":
            raise RuntimeError("selected base config is not linux/arm64")

        layer_tars = []
        for number, descriptor in enumerate(manifest["layers"]):
            blob = temporary_path / ("layer-%03d.blob" % number)
            print("downloading base layer %d/%d" % (number + 1, len(manifest["layers"])), flush=True)
            download_blob(descriptor["digest"], token, blob)
            layer_tars.append(decompress_layer(blob, descriptor.get("mediaType", "")))

        app_layer = make_app_layer(script_path)
        layer_tars.append(app_layer)
        app_diff_id = "sha256:" + hashlib.sha256(app_layer).hexdigest()
        config["rootfs"]["diff_ids"].append(app_diff_id)
        runtime = config.setdefault("config", {})
        runtime["WorkingDir"] = "/app"
        runtime["Cmd"] = ["python", "/app/smoke.py"]
        environment = [item for item in runtime.get("Env", []) if not item.startswith(("PYTHONDONTWRITEBYTECODE=", "PYTHONUNBUFFERED="))]
        environment.extend(("PYTHONDONTWRITEBYTECODE=1", "PYTHONUNBUFFERED=1"))
        runtime["Env"] = environment
        config.setdefault("history", []).append(
            {
                "created": "1970-01-01T00:00:00Z",
                "created_by": "COPY smoke.py /app/smoke.py",
                "comment": "vision runtime smoke layer",
            }
        )
        config_bytes = json_bytes(config)
        config_name = hashlib.sha256(config_bytes).hexdigest() + ".json"

        diff_ids = config["rootfs"]["diff_ids"]
        if len(diff_ids) != len(layer_tars):
            raise RuntimeError("base manifest layer count does not match config diff_ids")
        layer_ids = chain_ids(diff_ids)
        layer_paths = [layer_id + "/layer.tar" for layer_id in layer_ids]
        docker_manifest = [{"Config": config_name, "RepoTags": [IMAGE_TAG], "Layers": layer_paths}]
        repositories = {"vision-runtime": {"smoke": layer_ids[-1]}}

        temporary_output = output_path.with_name(output_path.name + ".tmp")
        try:
            with tarfile.open(temporary_output, mode="w") as archive:
                add_bytes(archive, config_name, config_bytes)
                add_bytes(archive, "manifest.json", json_bytes(docker_manifest))
                add_bytes(archive, "repositories", json_bytes(repositories))
                parent = None
                for layer_id, layer_data in zip(layer_ids, layer_tars):
                    layer_metadata = {"id": layer_id}
                    if parent is not None:
                        layer_metadata["parent"] = parent
                    add_bytes(archive, layer_id + "/VERSION", b"1.0")
                    add_bytes(archive, layer_id + "/json", json_bytes(layer_metadata))
                    add_bytes(archive, layer_id + "/layer.tar", layer_data)
                    parent = layer_id
            os.replace(str(temporary_output), str(output_path))
        finally:
            if temporary_output.exists():
                temporary_output.unlink()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    script_path = Path(__file__).with_name("smoke.py")
    build(args.output, script_path)
    print("created=%s" % args.output)


if __name__ == "__main__":
    main()
