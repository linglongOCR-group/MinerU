import inspect
import zipfile

import pytest
from fastapi import HTTPException

from mineru.backend.vlm import vlm_analyze
from mineru.cli import api_client, fast_api
from mineru.cli.public_http_client_policy import validate_public_http_client_request


def test_parse_request_form_data_includes_stage_urls_only_when_provided():
    form_data = api_client.build_parse_request_form_data(
        lang_list=["ch"],
        backend="vlm-http-client",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url="http://shared",
        layout_server_url="http://layout",
        recognition_server_url="http://recognition",
        start_page_id=0,
        end_page_id=None,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=False,
        response_format_zip=False,
        return_original_file=False,
        dissection_enable=True,
        stream=True,
    )

    assert form_data["server_url"] == "http://shared"
    assert form_data["layout_server_url"] == "http://layout"
    assert form_data["recognition_server_url"] == "http://recognition"
    assert form_data["dissection_enable"] == "true"
    assert form_data["stream"] == "true"

    form_data = api_client.build_parse_request_form_data(
        lang_list=["ch"],
        backend="vlm-http-client",
        parse_method="auto",
        formula_enable=True,
        table_enable=True,
        server_url=None,
        layout_server_url=None,
        recognition_server_url=None,
        start_page_id=0,
        end_page_id=None,
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=False,
        response_format_zip=False,
        return_original_file=False,
        dissection_enable=False,
        stream=False,
    )

    assert "server_url" not in form_data
    assert "layout_server_url" not in form_data
    assert "recognition_server_url" not in form_data
    assert form_data["dissection_enable"] == "false"
    assert form_data["stream"] == "false"


def test_fastapi_form_exposes_stage_url_fields():
    signature = inspect.signature(fast_api.parse_request_form)

    assert "layout_server_url" in signature.parameters
    assert "recognition_server_url" in signature.parameters
    assert "dissection_enable" in signature.parameters
    assert "stream" in signature.parameters
    assert "layout_server_url" in fast_api.ParseRequestOptions.__dataclass_fields__
    assert "recognition_server_url" in fast_api.AsyncParseTask.__dataclass_fields__
    assert "dissection_enable" in fast_api.ParseRequestOptions.__dataclass_fields__
    assert "dissection_enable" in fast_api.AsyncParseTask.__dataclass_fields__
    assert "stream" in fast_api.ParseRequestOptions.__dataclass_fields__
    assert "stream" in fast_api.AsyncParseTask.__dataclass_fields__


def test_result_zip_includes_dissection_directory_when_enabled(tmp_path):
    parse_dir = tmp_path / "doc" / "vlm"
    dissection_dir = parse_dir / "dissection"
    dissection_dir.mkdir(parents=True)
    (parse_dir / "doc.md").write_text("hello", encoding="utf-8")
    (dissection_dir / "manifest.json").write_text("{}", encoding="utf-8")
    (dissection_dir / "errors.jsonl").write_text("", encoding="utf-8")

    zip_path = fast_api.create_result_zip(
        output_dir=str(tmp_path),
        pdf_file_names=["doc"],
        backend="vlm-http-client",
        parse_method="auto",
        return_md=True,
        return_middle_json=False,
        return_model_output=False,
        return_content_list=False,
        return_images=False,
        return_original_file=False,
        dissection_enable=True,
    )

    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = set(zf.namelist())
    finally:
        fast_api.cleanup_file(zip_path)

    assert "doc/vlm/doc.md" in names
    assert "doc/vlm/dissection/manifest.json" in names
    assert "doc/vlm/dissection/errors.jsonl" in names


@pytest.mark.parametrize(
    "field",
    ["server_url", "layout_server_url", "recognition_server_url"],
)
def test_public_http_client_policy_blocks_all_caller_supplied_urls(field):
    kwargs = {
        "public_bind_exposed": True,
        "allow_public_http_client": False,
        "backend": "pipeline",
        "server_url": None,
        "layout_server_url": None,
        "recognition_server_url": None,
    }
    kwargs[field] = "http://example.test"

    with pytest.raises(HTTPException) as exc_info:
        validate_public_http_client_request(**kwargs)

    assert exc_info.value.status_code == 400


def test_model_singleton_cache_key_includes_stage_urls(monkeypatch):
    vlm_analyze.ModelSingleton().shutdown()
    created = []

    class FakeMinerUClient:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            created.append(self)

    monkeypatch.setattr(vlm_analyze, "MinerUClient", FakeMinerUClient)

    singleton = vlm_analyze.ModelSingleton()
    first = singleton.get_model(
        "http-client",
        None,
        "http://shared",
        "http://layout-a",
        "http://recognition",
    )
    second = singleton.get_model(
        "http-client",
        None,
        "http://shared",
        "http://layout-b",
        "http://recognition",
    )
    reused = singleton.get_model(
        "http-client",
        None,
        "http://shared",
        "http://layout-a",
        "http://recognition",
    )

    assert first is reused
    assert first is not second
    assert len(created) == 2
    assert created[0].kwargs["layout_server_url"] == "http://layout-a"
    assert created[1].kwargs["layout_server_url"] == "http://layout-b"

    singleton.shutdown()
