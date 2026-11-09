import httpx

from statpad_cbb.ingest.client import CbbdClient, unwrap_list


def test_client_sends_bearer_token_and_returns_directory_object():
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-token"
        assert request.url.params["season"] == "2026"
        assert request.url.path.endswith("/teams/directory")
        return httpx.Response(
            200,
            json={
                "season": 2026,
                "teams": [{"id": 1, "school": "Duke", "conferenceId": 10}],
                "conferences": [{"id": 10, "name": "ACC"}],
            },
        )

    transport = httpx.MockTransport(handler)
    client = CbbdClient(
        api_key="test-token",
        base_url="https://api.collegebasketballdata.com",
        min_request_interval_seconds=0,
        transport=transport,
        max_attempts=1,
    )
    payload = client.teams_directory(2026)
    assert isinstance(payload, dict)
    assert unwrap_list(payload, path="/teams/directory") == [
        {"id": 1, "school": "Duke", "conferenceId": 10}
    ]
    client.close()


def test_unwrap_list_supports_common_envelopes():
    assert unwrap_list([1, 2]) == [1, 2]
    assert unwrap_list({"teams": [{"id": 1}]}) == [{"id": 1}]
    assert unwrap_list(None) == []
