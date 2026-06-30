from fastapi.testclient import TestClient

from app.main import app
from app.schemas import ChatResponse, RecommendationItem


def test_health_returns_ok():
    with TestClient(app) as client:
        resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_chat_returns_exact_schema_shape(monkeypatch):
    fake = ChatResponse(
        reply="hi",
        recommendations=[RecommendationItem(name="X", url="https://example.com/x", test_type="K")],
        end_of_conversation=False,
    )
    monkeypatch.setattr("app.main.handle_chat", lambda messages: fake)

    with TestClient(app) as client:
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {"reply", "recommendations", "end_of_conversation"}
    assert body["recommendations"] == [{"name": "X", "url": "https://example.com/x", "test_type": "K"}]


def test_chat_empty_recommendations_serializes_as_empty_array(monkeypatch):
    fake = ChatResponse(reply="clarify please", recommendations=[], end_of_conversation=False)
    monkeypatch.setattr("app.main.handle_chat", lambda messages: fake)

    with TestClient(app) as client:
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "I need an assessment"}]})

    assert resp.json()["recommendations"] == []


def test_chat_falls_back_gracefully_on_pipeline_exception(monkeypatch):
    def boom(messages):
        raise RuntimeError("simulated pipeline failure")

    monkeypatch.setattr("app.main.handle_chat", boom)

    with TestClient(app) as client:
        resp = client.post("/chat", json={"messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 200
    body = resp.json()
    assert body["recommendations"] == []
    assert isinstance(body["reply"], str) and body["reply"]


def test_chat_rejects_malformed_role():
    with TestClient(app) as client:
        resp = client.post("/chat", json={"messages": [{"role": "system", "content": "bad role"}]})
    assert resp.status_code == 422


def test_chat_rejects_missing_messages_field():
    with TestClient(app) as client:
        resp = client.post("/chat", json={})
    assert resp.status_code == 422
