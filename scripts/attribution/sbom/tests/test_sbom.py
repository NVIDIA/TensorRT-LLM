import json
from pathlib import Path

import pytest
from click.testing import CliRunner
from jsonschema import Draft7Validator
from trtllm_sbom.cli import main


@pytest.fixture(scope="session")
def cyclonedx_schema() -> dict:
    schema_path = Path(__file__).resolve().parent / "schemas" / "bom-1.6.schema.json"
    return json.loads(schema_path.read_text(encoding="utf-8"))


def test_cyclonedx_json_schema_validation(
    tmp_path: Path, stub_requests_head, cyclonedx_schema: dict
):
    runner = CliRunner()
    data_dir = tmp_path / "scripts/attribution/data"

    # Minimal data to generate SBOM
    lic_path = tmp_path / "LICENSE"
    lic_path.write_text("MIT\nCopyright 2024", encoding="utf-8")
    res = runner.invoke(
        main,
        [
            "import",
            "--data-dir",
            str(data_dir),
            "-d",
            "tensorrt",
            "-v",
            "10.14.0.26+abcdef0123456789abcdef0123456789abcdef01",
            "-s",
            "https://github.com/NVIDIA/TensorRT/commit/abcdef0123456789abcdef0123456789abcdef01",
            "-l",
            str(lic_path),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    f = tmp_path / "include" / "x.h"
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text("// hdr\n", encoding="utf-8")
    res = runner.invoke(
        main,
        [
            "map-files",
            "--data-dir",
            str(data_dir),
            "-d",
            "tensorrt",
            "-v",
            "10.14.0.26+abcdef0123456789abcdef0123456789abcdef01",
            str(f),
        ],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output

    sbom = tmp_path / "sbom.json"
    res = runner.invoke(
        main,
        ["generate", "--data-dir", str(data_dir), "-s", str(sbom), str(f)],
        catch_exceptions=False,
    )
    assert res.exit_code == 0, res.output
    assert sbom.exists()

    # Validate against CycloneDX JSON schema
    data = json.loads(sbom.read_text(encoding="utf-8"))
    Draft7Validator(cyclonedx_schema).validate(data)
