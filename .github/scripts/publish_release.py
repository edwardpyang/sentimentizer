"""Create the version tag and GitHub release after a successful PyPI publish."""

import json
import os
import re
import tomllib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import Request, urlopen

GITHUB_API_VERSION = "2026-03-10"
HTTP_TIMEOUT_S = 30
MAX_TAG_DEPTH = 10


def _github_request(
    method: str, path: str, payload: dict[str, object] | None = None
) -> dict[str, object] | None:
    """Call GitHub with bounded HTTP work; only a missing GET resource is optional."""
    repository = os.environ["GITHUB_REPOSITORY"]
    request = Request(
        f"https://api.github.com/repos/{repository}/{path}",
        data=json.dumps(payload).encode() if payload is not None else None,
        headers={
            "Authorization": f"Bearer {os.environ['GH_TOKEN']}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
            "X-GitHub-Api-Version": GITHUB_API_VERSION,
        },
        method=method,
    )
    try:
        with urlopen(request, timeout=HTTP_TIMEOUT_S) as response:
            result = json.load(response)
    except HTTPError as error:
        if method == "GET" and error.code == 404:
            return None
        raise
    except (TimeoutError, URLError) as error:
        if isinstance(error, TimeoutError) or isinstance(error.reason, TimeoutError):
            raise TimeoutError(
                f"GitHub {method} {path} timed out after {HTTP_TIMEOUT_S}s"
            ) from error
        raise
    if not isinstance(result, dict):
        raise ValueError(f"Invalid GitHub response for {method} {path}: expected an object")
    return result


def _tag_commit(reference: dict[str, object]) -> str:
    """Resolve lightweight and annotated tags without trusting their response shape."""
    target = reference.get("object")
    for _ in range(MAX_TAG_DEPTH):
        if not isinstance(target, dict):
            break
        sha = target.get("sha")
        if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-fA-F]{40}", sha):
            break
        if target.get("type") == "commit":
            return sha.lower()
        if target.get("type") != "tag":
            break
        tagged = _github_request("GET", f"git/tags/{sha}")
        target = tagged.get("object") if tagged is not None else None
    raise ValueError("Version tag does not resolve to a commit")


def ensure_release(version: str, commit: str) -> None:
    """Reuse matching resources or create them, never moving an existing version tag."""
    if not isinstance(version, str) or not re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+", version):
        raise ValueError("Expected a stable semantic package version, such as 0.333.0")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", commit):
        raise ValueError("Expected the full published commit SHA")
    tag_path = f"git/ref/tags/{quote(version, safe='')}"
    reference = _github_request("GET", tag_path)
    if reference is None:
        try:
            reference = _github_request(
                "POST", "git/refs", {"ref": f"refs/tags/{version}", "sha": commit}
            )
        except HTTPError as error:
            if error.code != 422:
                raise
            reference = _github_request("GET", tag_path)
            if reference is None:
                raise
    if reference is None or _tag_commit(reference) != commit.lower():
        raise ValueError(
            f"Tag {version} does not point at published commit {commit}; refusing to move it"
        )

    release_path = f"releases/tags/{quote(version, safe='')}"
    release = _github_request("GET", release_path)
    if release is None:
        try:
            release = _github_request(
                "POST",
                "releases",
                {
                    "tag_name": version,
                    "target_commitish": commit,
                    "name": version,
                    "draft": False,
                    "prerelease": False,
                    "generate_release_notes": True,
                    "make_latest": "legacy",
                },
            )
        except HTTPError as error:
            if error.code != 422:
                raise
            release = _github_request("GET", release_path)
            if release is None:
                raise
    if release is None or release.get("tag_name") != version or release.get("draft") is not False:
        raise ValueError(f"GitHub release for {version} is missing, malformed, or still a draft")


def main() -> None:
    """Use the checked-out package version and the exact commit selected by CD."""
    with Path("pyproject.toml").open("rb") as stream:
        version = tomllib.load(stream)["project"]["version"]
    commit = os.environ["GITHUB_SHA"]
    ensure_release(version, commit)
    print(f"GitHub release {version} is ready at commit {commit}.")


if __name__ == "__main__":
    main()
