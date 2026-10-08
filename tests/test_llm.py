from types import SimpleNamespace
from unittest.mock import Mock

from src.llm import LLMClient
from src.models import ResumeAnalysis


def test_gpt_oss_request_reserves_tokens_for_json_output():
    client = LLMClient.__new__(LLMClient)
    client.client = Mock()
    client.client.chat.completions.create.return_value = SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content='{"ai_project_depth_score": 0}')
            )
        ]
    )

    result = client._call_model("Evaluate this candidate.")

    request = client.client.chat.completions.create.call_args.kwargs
    assert result == '{"ai_project_depth_score": 0}'
    assert request["max_tokens"] == 4096
    assert request["reasoning_effort"] == "low"
    assert request["response_format"]["type"] == "json_schema"
    assert (
        request["response_format"]["json_schema"]["schema"]
        == ResumeAnalysis.model_json_schema()
    )