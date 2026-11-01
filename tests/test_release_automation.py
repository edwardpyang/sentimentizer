"""Exercise CD tag/release creation with HTTP mocked at the GitHub boundary."""

import importlib.util
import io
import json
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / ".github/scripts/publish_release.py"
VERSION = "0.333.0"
COMMIT = "a" * 40
OTHER_COMMIT = "b" * 40
REFERENCE = {"object": {"type": "commit", "sha": COMMIT}}
RELEASE = {"tag_name": VERSION, "draft": False}


@pytest.fixture
def api(monkeypatch):
    spec = importlib.util.spec_from_file_location("publish_release", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    replies = deque()
    calls = []

    def urlopen(request, timeout):
        assert timeout == module.HTTP_TIMEOUT_S
        assert request.get_header("Authorization") == "Bearer test-token"
        path = request.full_url.removeprefix("https://api.github.com/repos/owner/repo/")
        payload = json.loads(request.data) if request.data is not None else None
        calls.append((request.method, path, payload))
        assert replies, f"Unexpected API request: {request.method} {path}"
        result = replies.popleft()
        if isinstance(result, int):
            raise HTTPError(request.full_url, result, "GitHub error", {}, None)
        if isinstance(result, Exception):
            raise result
        return io.BytesIO(json.dumps(result).encode())

    monkeypatch.setattr(module, "urlopen", urlopen)
    monkeypatch.setenv("GITHUB_REPOSITORY", "owner/repo")
    monkeypatch.setenv("GH_TOKEN", "test-token")
    return SimpleNamespace(module=module, replies=replies, calls=calls)


def test_success_creates_bare_version_tag_at_published_commit_and_generated_release(api):
    api.replies.extend([404, REFERENCE, 404, RELEASE])
    api.module.ensure_release(VERSION, COMMIT)
    assert api.calls == [
        ("GET", f"git/ref/tags/{VERSION}", None),
        ("POST", "git/refs", {"ref": f"refs/tags/{VERSION}", "sha": COMMIT}),
        ("GET", f"releases/tags/{VERSION}", None),
        (
            "POST",
            "releases",
            {
                "tag_name": VERSION,
                "target_commitish": COMMIT,
                "name": VERSION,
                "draft": False,
                "prerelease": False,
                "generate_release_notes": True,
                "make_latest": "legacy",
            },
        ),
    ]
    assert not api.replies


def test_rerun_reuses_existing_tag_and_release_without_writing(api):
    api.replies.extend([REFERENCE, RELEASE])
    api.module.ensure_release(VERSION, COMMIT)
    assert all(method == "GET" for method, _, _ in api.calls)


def test_annotated_tag_resolves_to_published_commit(api):
    annotated = {"object": {"type": "tag", "sha": OTHER_COMMIT}}
    api.replies.extend([annotated, REFERENCE, RELEASE])
    api.module.ensure_release(VERSION, COMMIT)
    assert api.calls[1] == ("GET", f"git/tags/{OTHER_COMMIT}", None)
    assert all(method == "GET" for method, _, _ in api.calls)


def test_existing_tag_at_another_commit_is_never_moved(api):
    api.replies.append({"object": {"type": "commit", "sha": OTHER_COMMIT}})
    with pytest.raises(ValueError, match="refusing to move"):
        api.module.ensure_release(VERSION, COMMIT)
    assert len(api.calls) == 1


def test_failed_release_creation_leaves_tag_reusable_on_retry(api):
    api.replies.extend([404, REFERENCE, 404, 503])
    with pytest.raises(HTTPError) as error:
        api.module.ensure_release(VERSION, COMMIT)
    assert error.value.code == 503
    api.calls.clear()
    api.replies.extend([REFERENCE, 404, RELEASE])
    api.module.ensure_release(VERSION, COMMIT)
    assert [path for method, path, _ in api.calls if method == "POST"] == ["releases"]


@pytest.mark.parametrize("resource", ["tag", "release"])
def test_concurrent_creation_rechecks_the_existing_resource(api, resource):
    if resource == "tag":
        api.replies.extend([404, 422, REFERENCE, RELEASE])
    else:
        api.replies.extend([REFERENCE, 404, 422, RELEASE])
    api.module.ensure_release(VERSION, COMMIT)
    assert not api.replies


@pytest.mark.parametrize("resource", ["tag", "release"])
def test_validation_error_without_an_existing_resource_stays_a_failure(api, resource):
    if resource == "tag":
        api.replies.extend([404, 422, 404])
    else:
        api.replies.extend([REFERENCE, 404, 422, 404])
    with pytest.raises(HTTPError) as error:
        api.module.ensure_release(VERSION, COMMIT)
    assert error.value.code == 422


def test_concurrently_created_conflicting_tag_is_rejected(api):
    api.replies.extend([404, 422, {"object": {"type": "commit", "sha": OTHER_COMMIT}}])
    with pytest.raises(ValueError, match="refusing to move"):
        api.module.ensure_release(VERSION, COMMIT)
    assert all(path != "releases" for _, path, _ in api.calls)


@pytest.mark.parametrize("status", [401, 403, 500])
def test_lookup_errors_are_not_mistaken_for_missing_tags(api, status):
    api.replies.append(status)
    with pytest.raises(HTTPError) as error:
        api.module.ensure_release(VERSION, COMMIT)
    assert error.value.code == status
    assert "test-token" not in str(error.value)
    assert len(api.calls) == 1


@pytest.mark.parametrize("response", [None, [], "bad", {"object": None}, {"object": {"sha": 3}}])
def test_malformed_tag_responses_fail_before_creating_a_release(api, response):
    api.replies.append(response)
    with pytest.raises(ValueError):
        api.module.ensure_release(VERSION, COMMIT)
    assert len(api.calls) == 1


def test_annotated_tag_cycles_are_bounded(api, monkeypatch):
    monkeypatch.setattr(api.module, "MAX_TAG_DEPTH", 2)
    annotation = {"object": {"type": "tag", "sha": OTHER_COMMIT}}
    api.replies.extend([annotation] * 3)
    with pytest.raises(ValueError, match="resolve to a commit"):
        api.module.ensure_release(VERSION, COMMIT)
    assert len(api.calls) == 3


@pytest.mark.parametrize(
    "response", [{}, {"tag_name": "other", "draft": False}, {"tag_name": VERSION, "draft": True}]
)
def test_malformed_or_draft_releases_do_not_claim_success(api, response):
    api.replies.extend([REFERENCE, response])
    with pytest.raises(ValueError, match="missing, malformed, or still a draft"):
        api.module.ensure_release(VERSION, COMMIT)


@pytest.mark.parametrize("error", [TimeoutError(), URLError(TimeoutError())])
def test_timeouts_include_the_operation_and_budget(api, error):
    api.replies.append(error)
    with pytest.raises(TimeoutError, match=r"GET git/ref/tags/0.333.0 timed out after 30s"):
        api.module.ensure_release(VERSION, COMMIT)


def test_main_reads_checked_out_version_and_workflow_commit(api, tmp_path, monkeypatch, capsys):
    (tmp_path / "pyproject.toml").write_text(f'[project]\nversion = "{VERSION}"\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_SHA", COMMIT)
    api.replies.extend([404, REFERENCE, 404, RELEASE])
    api.module.main()
    assert VERSION in capsys.readouterr().out
    assert api.calls[1][2]["sha"] == COMMIT


@pytest.mark.parametrize("version,commit", [("../bad", COMMIT), (VERSION, "main")])
def test_invalid_version_or_commit_fails_before_network_calls(api, version, commit):
    with pytest.raises(ValueError):
        api.module.ensure_release(version, commit)
    assert not api.calls
