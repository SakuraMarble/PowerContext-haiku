# Copyright (c) 2026 OceanBase.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Release-blocking regressions, not evidence that any database upgrade works."""

import json
import os
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pytest
import upgrade_release_gate as gate
import upgrade_release_matrix as matrix
import yaml
from ci_legacy_upgrade_rejection import database_environment

from powercontext.server.settings import ServerSettings


@pytest.fixture
def release(tmp_path, monkeypatch):
    persistence = tmp_path / "src/powercontext/builtin/persistence/tables.py"
    persistence.parent.mkdir(parents=True)
    persistence.write_text("# disposable persistence input\n")
    contract = tmp_path / "openapi/powercontext.yaml"
    contract.parent.mkdir()
    contract.write_text("info:\n  version: 2.0.0\n")
    (tmp_path / "release").mkdir()
    policy = {
        "format": 2,
        "target_version": "2.0.0",
        "mode": "upgrade-required",
        "schema_review": {"persistence_sha256": gate.managed_digest(tmp_path), "rationale": "No upgrades supported"},
        "cases": [],
        "automatic_sources": None,
    }
    suite = tmp_path / "tests/release_upgrades/test_example.py"
    suite.parent.mkdir(parents=True)
    suite.write_text("# Gate orchestration fixture; not real upgrade acceptance.\n")
    automatic_policy(policy)
    monkeypatch.setattr(gate, "run_case", lambda _root, _wheel, case, _output: {"case": case, "status": "passed"})
    (tmp_path / gate.MANIFEST).write_text(json.dumps(policy))
    monkeypatch.setattr(
        matrix,
        "fetch_catalog",
        lambda: {
            "releases": {
                "1.3.0": [
                    {
                        "filename": "powercontext-1.3.0-py3-none-any.whl",
                        "packagetype": "bdist_wheel",
                        "url": "https://files.pythonhosted.org/packages/test.whl",
                        "digests": {"sha256": "a" * 64},
                    }
                ]
            }
        },
    )
    for language in ("en", "zh"):
        page = tmp_path / f"docs/{language}/docs/get-started/install-and-run.md"
        page.parent.mkdir(parents=True)
        page.write_text("<!-- upgrade-support:start -->\nplaceholder\n<!-- upgrade-support:end -->\n")
    gate.check_docs(tmp_path, policy, write=True)
    wheel = tmp_path / "powercontext-2.0.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("powercontext-2.0.0.dist-info/METADATA", "Name: powercontext\nVersion: 2.0.0\n")
    return tmp_path, wheel, policy


def test_required_mode_executes_three_cases_but_cannot_publish_as_supported(release):
    root, wheel, policy = release
    with pytest.raises(gate.GateError, match="support declaration"):
        gate.gate(root, wheel, root / "evidence", release=True, release_notes=gate.support_text(policy, "en"))
    result = json.loads((root / "evidence/result.json").read_text())
    assert result["status"] == "failed"
    assert len(result["cases"]) == 3
    assert all(case["status"] == "passed" for case in result["cases"])
    assert result["target_wheel_sha256"] == gate.sha256(wheel)


def test_release_notes_cannot_omit_the_support_limit(release):
    root, wheel, _ = release
    with pytest.raises(gate.GateError, match="notes"):
        gate.gate(root, wheel, root / "evidence", release=True, release_notes="Upgrade your deployment now.")


def test_release_rejects_wrong_artifact_version(release):
    root, wheel, _ = release
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("powercontext-9.0.0.dist-info/METADATA", "Name: powercontext\nVersion: 9.0.0\n")
    with pytest.raises(gate.GateError, match="Wheel version"):
        gate.gate(root, wheel, root / "evidence", release=True)


def test_schema_change_needs_a_new_compatibility_review(release):
    root, _, _ = release
    (root / "src/powercontext/builtin/persistence/tables.py").write_text("# different schema\n")
    with pytest.raises(gate.GateError, match="Persistence changed"):
        gate.load_policy(root)


def test_supported_mode_cannot_succeed_with_an_empty_matrix(release):
    root, _, policy = release
    policy["mode"] = "supported-upgrades"
    policy["automatic_sources"] = None
    (root / gate.MANIFEST).write_text(json.dumps(policy))
    with pytest.raises(gate.GateError, match="nonempty matrix"):
        gate.load_policy(root)


@pytest.mark.parametrize(
    "relative",
    [
        "src/powercontext/server/identity.py",
        "src/powercontext/server/authz/repository.py",
        "src/powercontext/builtin/code/seekdb.py",
        "src/powercontext/limits.py",
    ],
)
def test_schema_owners_outside_persistence_need_compatibility_review(release, relative):
    root, _, _ = release
    owner = root / relative
    owner.parent.mkdir(parents=True, exist_ok=True)
    owner.write_text("# changed managed schema input\n")
    with pytest.raises(gate.GateError, match="Persistence changed"):
        gate.load_policy(root)


def supported_case(root, policy):
    policy["mode"] = "supported-upgrades"
    policy["automatic_sources"] = None
    policy["cases"] = [
        {
            "source_version": "1.3.0",
            "backend": backend,
            "configuration": "full-server",
            "suite": "tests/release_upgrades/test_example.py",
        }
        for backend in ("sqlite", "seekdb", "oceanbase")
    ]
    (root / gate.MANIFEST).write_text(json.dumps(policy))
    gate.check_docs(root, policy, write=True)


@pytest.mark.parametrize(
    "blocker", ["migration_required", "unmanaged full Server layout", "migration_framework_not_ready"]
)
def test_any_upgrade_stage_failure_blocks_release_and_records_failure(release, monkeypatch, blocker):
    root, wheel, policy = release
    supported_case(root, policy)

    def fail(*_args):
        raise gate.GateError(blocker)

    monkeypatch.setattr(gate, "run_case", fail)
    with pytest.raises(gate.GateError, match="Required upgrade acceptance failed"):
        gate.gate(root, wheel, root / "evidence")
    evidence = json.loads((root / "evidence/result.json").read_text())
    assert evidence["status"] == "failed"
    assert len(evidence["cases"]) == 3
    assert all(case["status"] == "failed" and blocker in case["error"] for case in evidence["cases"])


def test_missing_suite_cannot_be_advertised(release):
    root, _, policy = release
    supported_case(root, policy)
    (root / policy["cases"][0]["suite"]).unlink()
    with pytest.raises(gate.GateError, match="Missing acceptance suite"):
        gate.load_policy(root)


def test_installation_guidance_cannot_drift_from_matrix(release):
    root, _, policy = release
    page = root / "docs/en/docs/get-started/install-and-run.md"
    page.write_text(page.read_text().replace(gate.NOTICE["en"], "All upgrades are supported."))
    with pytest.raises(gate.GateError, match="instructions differ"):
        gate.check_docs(root, policy)


@pytest.mark.parametrize("invalid", ["empty", "missing-startup", "skipped", "failure", "error", "truncated"])
def test_partial_or_skipped_acceptance_is_not_passing_evidence(tmp_path, invalid):
    report = tmp_path / "junit.xml"
    suite = ElementTree.Element("testsuite")
    for phase in sorted(gate.PHASES):
        if invalid == "empty" or (invalid == "missing-startup" and phase == "test_target_startup"):
            continue
        case = ElementTree.SubElement(suite, "testcase", name=phase)
        if phase == "test_target_startup" and invalid in {"skipped", "failure", "error"}:
            ElementTree.SubElement(case, invalid)
    ElementTree.ElementTree(suite).write(report)
    if invalid == "truncated":
        report.write_text("<testsuite>")
    with pytest.raises((gate.GateError, ElementTree.ParseError)):
        gate.validate_junit(report)


def test_complete_passing_report_is_accepted(tmp_path):
    report = tmp_path / "junit.xml"
    suite = ElementTree.Element("testsuite")
    for phase in gate.PHASES:
        ElementTree.SubElement(suite, "testcase", name=phase)
    ElementTree.ElementTree(suite).write(report)
    gate.validate_junit(report)


def test_missing_backend_evidence_fails_even_with_a_passing_report(tmp_path):
    report = tmp_path / "backend.json"
    report.write_text(json.dumps({"backend": "oceanbase", "engine_version": "4.6.3"}))
    with pytest.raises(gate.GateError, match="driver_version"):
        gate.validate_backend_evidence(report, "oceanbase")


def test_old_evidence_directory_cannot_be_reused(release):
    root, wheel, _ = release
    gate.gate(root, wheel, root / "evidence")
    with pytest.raises(FileExistsError):
        gate.gate(root, wheel, root / "evidence")


def catalog_for(*versions):
    return {
        "releases": {
            version: [
                {
                    "filename": f"powercontext-{version}-py3-none-any.whl",
                    "packagetype": "bdist_wheel",
                    "yanked": False,
                    "url": f"https://files.pythonhosted.org/packages/powercontext-{version}-py3-none-any.whl",
                    "digests": {"sha256": "a" * 64},
                }
            ]
            for version in versions
        }
    }


def automatic_policy(policy):
    policy["automatic_sources"] = {
        "strategy": "fixed-version",
        "source_version": "1.3.0",
        "profiles": [
            {
                "backend": backend,
                "configuration": "full-server-no-vector",
                "suite": "tests/release_upgrades/test_example.py",
            }
            for backend in ("sqlite", "seekdb", "oceanbase")
        ],
    }


@pytest.mark.parametrize("target", ["1.4.0", "1.4.1", "1.5.0", "1.5.1.dev2"])
def test_fixed_source_tracks_each_target_on_all_three_backends(release, monkeypatch, target):
    _, _, policy = release
    automatic_policy(policy)
    monkeypatch.setattr(
        matrix, "fetch_catalog", lambda: catalog_for("1.3.0", "1.4.0", "1.4.1", "1.5.0rc1", "1.5.0", "1.6.0")
    )
    resolved = matrix.resolve_matrix(policy, target)
    assert [(case["source_version"], case["backend"]) for case in resolved["candidate_cases"]] == [
        ("1.3.0", backend) for backend in ("sqlite", "seekdb", "oceanbase")
    ]
    assert resolved["required_cases"] == resolved["candidate_cases"]
    assert all(case["source_artifact"]["sha256"] == "a" * 64 for case in resolved["candidate_cases"])


def test_newly_published_versions_do_not_change_the_fixed_source(release, monkeypatch):
    _, _, policy = release
    automatic_policy(policy)
    catalog = catalog_for("1.3.0")
    monkeypatch.setattr(matrix, "fetch_catalog", lambda: catalog)
    first = matrix.resolve_matrix(policy, "1.5.0")
    catalog["releases"].update(catalog_for("1.4.1")["releases"])
    second = matrix.resolve_matrix(policy, "1.5.0")
    assert first["candidate_cases"] == second["candidate_cases"]
    assert len(second["candidate_cases"]) == 3


@pytest.mark.parametrize("target", ["1.2.0", "1.3.0", "1.3.0rc1"])
def test_fixed_source_must_precede_target(release, target):
    _, _, policy = release
    automatic_policy(policy)
    with pytest.raises(ValueError, match="must precede target"):
        matrix.resolve_matrix(policy, target)


@pytest.mark.parametrize("broken", ["empty", "missing-source", "yanked", "missing-wheel", "bad-hash", "foreign-origin"])
def test_unavailable_source_cannot_silently_shrink_matrix(release, monkeypatch, broken):
    _, _, policy = release
    automatic_policy(policy)
    catalog = catalog_for("1.3.0", "1.4.1")
    wheel = catalog["releases"]["1.3.0"][0]
    if broken == "empty":
        catalog["releases"] = {}
    elif broken == "missing-source":
        del catalog["releases"]["1.3.0"]
    elif broken == "yanked":
        wheel["yanked"] = True
    elif broken == "missing-wheel":
        catalog["releases"]["1.3.0"] = []
    elif broken == "bad-hash":
        wheel["digests"]["sha256"] = "invalid"
    else:
        wheel["url"] = "https://untrusted.invalid/package.whl"
    monkeypatch.setattr(matrix, "fetch_catalog", lambda: catalog)
    with pytest.raises(ValueError):
        matrix.resolve_matrix(policy, "1.5.0")


def test_catalog_outage_records_a_failed_gate(release, monkeypatch):
    root, wheel, policy = release
    automatic_policy(policy)
    (root / gate.MANIFEST).write_text(json.dumps(policy))

    def unavailable():
        raise OSError("Catalog unavailable")  # noqa: TRY003 -- exercise a distinct discovery outage

    monkeypatch.setattr(matrix, "fetch_catalog", unavailable)
    with pytest.raises(OSError, match="Catalog unavailable"):
        gate.gate(root, wheel, root / "evidence")
    assert json.loads((root / "evidence/result.json").read_text())["status"] == "failed"


def test_switching_to_supported_upgrades_requires_all_backend_suites(release):
    root, _, policy = release
    automatic_policy(policy)
    policy["automatic_sources"]["profiles"][0]["suite"] = None
    (root / gate.MANIFEST).write_text(json.dumps(policy))
    with pytest.raises(gate.GateError, match="every backend profile"):
        gate.load_policy(root)


def test_supported_fixed_source_executes_every_backend(release, monkeypatch):
    root, wheel, policy = release
    supported_case(root, policy)
    suite = policy["cases"][0]["suite"]
    policy["cases"] = []
    automatic_policy(policy)
    for profile in policy["automatic_sources"]["profiles"]:
        profile["suite"] = suite
    (root / gate.MANIFEST).write_text(json.dumps(policy))
    gate.check_docs(root, policy, write=True)
    monkeypatch.setattr(matrix, "fetch_catalog", lambda: catalog_for("1.3.0", "1.4.1"))
    executed = []

    def execute(_root, _wheel, case, _output):
        executed.append((case["source_version"], case["backend"]))
        return {"case": case, "status": "passed"}

    monkeypatch.setattr(gate, "run_case", execute)
    gate.gate(root, wheel, root / "evidence")
    assert executed == [(version, backend) for version in ("1.3.0",) for backend in ("sqlite", "seekdb", "oceanbase")]
    result = json.loads((root / "evidence/result.json").read_text())
    assert len(result["cases"]) == len(result["resolved_matrix"]["required_cases"]) == 3


def test_publishing_jobs_depend_on_nonoptional_preupload_upgrade_gate():
    root = Path(__file__).parents[1]
    workflow = yaml.safe_load((root / ".github/workflows/release.yml").read_text())
    build = workflow["jobs"]["release-build"]
    steps = build["steps"]
    index = next(i for i, step in enumerate(steps) if "run_upgrade_release_gate.py" in step.get("run", ""))
    assert "if" not in steps[index] and not steps[index].get("continue-on-error")
    assert "--release --release-event" in steps[index]["run"]
    upload = next(i for i, step in enumerate(steps) if step.get("with", {}).get("name") == "release-dists")
    assert index < upload
    assert steps[upload]["with"]["path"] == "${{ runner.temp }}/upgrade-release-evidence/validated-dist/*"
    for name in ("pypi-publish", "release-assets"):
        assert workflow["jobs"][name]["needs"] == "release-build"


def test_repository_baseline_covers_all_three_databases():
    policy = gate.load_policy(Path(__file__).parents[1])
    selection = policy["automatic_sources"]
    assert selection["strategy"] == "fixed-version"
    assert selection["source_version"] == "1.3.0"
    assert {profile["backend"] for profile in selection["profiles"]} == {"sqlite", "seekdb", "oceanbase"}


@pytest.mark.parametrize("workflow_name,job_name", [("master", "upgrade-release-gate"), ("release", "release-build")])
def test_upgrade_jobs_require_real_backends_and_always_clean_up(workflow_name, job_name):
    root = Path(__file__).parents[1]
    workflow = yaml.safe_load((root / f".github/workflows/{workflow_name}.yml").read_text())
    job = workflow["jobs"][job_name]
    steps = job["steps"]
    setup_index = next(
        i for i, step in enumerate(steps) if step.get("uses") == "./.github/actions/setup-upgrade-oceanbase"
    )
    gate_index = next(i for i, step in enumerate(steps) if "run_upgrade_release_gate.py" in step.get("run", ""))
    assert setup_index < gate_index
    for index in (setup_index, gate_index):
        assert "if" not in steps[index] and not steps[index].get("continue-on-error")
    assert not any("ci_legacy_upgrade_rejection.py" in step.get("run", "") for step in steps)
    assert "/pc_upgrade_test?" in job["env"]["PC_UPGRADE_OCEANBASE_URL"]
    assert "/pc_upgrade_fresh?" in job["env"]["PC_UPGRADE_OCEANBASE_FRESH_URL"]
    cleanup = next(step for step in steps if "docker rm --force upgrade-oceanbase" in step.get("run", ""))
    assert cleanup["if"] == "always()"


def test_oceanbase_fixture_selects_the_oceanbase_server_profile(tmp_path, monkeypatch):
    for key in os.environ:
        if key.startswith("POWERCONTEXT_"):
            monkeypatch.delenv(key)
    url = "mysql+aoceanbase://root%40test:test@127.0.0.1:2881/pc_upgrade_test?charset=utf8mb4"
    for key, value in database_environment("oceanbase", tmp_path / "unused", {"PC_UPGRADE_OCEANBASE_URL": url}).items():
        monkeypatch.setenv(key, value)
    settings = ServerSettings()
    assert settings.database.kind == "oceanbase"
    assert settings.database.url.get_secret_value() == url


@pytest.mark.parametrize("mode", ["fresh-install-only", "upgrade-required", "supported-upgrades"])
def test_declarations_cannot_bypass_missing_backend_or_suite(release, mode):
    root, _, policy = release
    policy["mode"] = mode
    policy["automatic_sources"]["profiles"] = policy["automatic_sources"]["profiles"][:2]
    (root / gate.MANIFEST).write_text(json.dumps(policy))
    with pytest.raises(gate.GateError):
        gate.load_policy(root)


def test_fresh_install_only_is_not_an_upgrade_escape_hatch(release):
    root, _, policy = release
    policy["mode"] = "fresh-install-only"
    (root / gate.MANIFEST).write_text(json.dumps(policy))
    with pytest.raises(gate.GateError, match="mode"):
        gate.load_policy(root)


def test_failed_backend_does_not_skip_other_required_backends(release, monkeypatch):
    root, wheel, _ = release
    executed = []

    def execute(_root, _wheel, case, _output):
        executed.append(case["backend"])
        if case["backend"] == "sqlite":
            raise gate.GateError("migration_required")
        return {"case": case, "status": "passed"}

    monkeypatch.setattr(gate, "run_case", execute)
    with pytest.raises(gate.GateError, match="failed: sqlite"):
        gate.gate(root, wheel, root / "evidence")
    assert executed == ["sqlite", "seekdb", "oceanbase"]
    result = json.loads((root / "evidence/result.json").read_text())
    assert result["status"] == "failed"
    assert [case["status"] for case in result["cases"]] == ["failed", "passed", "passed"]


@pytest.mark.parametrize(
    "relative",
    [
        "src/powercontext/builtin/persistence/tables.py",
        "src/powercontext/builtin/persistence/revision.sql",
        "src/powercontext/builtin/code/layout.json",
        "src/powercontext/server/authz/repository.py",
        "src/powercontext/server/identity.py",
        "src/powercontext/limits.py",
    ],
)
def test_checkout_line_endings_preserve_compatibility_review(release, relative):
    root, _, policy = release
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"first line\nsecond line\n")
    policy["schema_review"]["persistence_sha256"] = gate.managed_digest(root)
    (root / gate.MANIFEST).write_text(json.dumps(policy))
    path.write_bytes(b"first line\r\nsecond line\r\n")
    assert gate.load_policy(root)["schema_review"] == policy["schema_review"]
    path.write_bytes(b"first line\r\nchanged content\r\n")
    with pytest.raises(gate.GateError, match="file set or content; CRLF/LF normalized"):
        gate.load_policy(root)


def test_artifact_hashes_do_not_normalize_bytes(tmp_path):
    artifact = tmp_path / "artifact"
    artifact.write_bytes(b"payload\n")
    before = gate.sha256(artifact)
    artifact.write_bytes(b"payload\r\n")
    assert gate.sha256(artifact) != before
