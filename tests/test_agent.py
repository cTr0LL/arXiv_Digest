"""Tests for the hand-rolled agent loop.

A fake client returns scripted responses, so the loop's mechanics -- message
threading, tool_use_id pairing, stop_reason handling, the iteration ceiling --
are all exercised without an API key or a cent of spend.

The message-threading tests matter more than they look. Splitting tool results
across several user messages, or forgetting to append the assistant turn, both
produce a loop that still runs and still returns answers, just worse ones. They
fail silently, which is exactly what tests are for.
"""

from __future__ import annotations

import pytest

from arxiv_digest import agent
from conftest import make_paper


class Block:
    def __init__(self, type, id=None, name=None, input=None, text=None):
        self.type = type
        self.id = id
        self.name = name
        self.input = input or {}
        self.text = text


class Response:
    def __init__(self, content, stop_reason="tool_use"):
        self.content = content
        self.stop_reason = stop_reason
        self.stop_details = None


class FakeMessages:
    def __init__(self, script, calls):
        self._script = list(script)
        self._index = 0
        self.calls = calls

    def create(self, **kwargs):
        # Snapshot the message list. The loop mutates one list in place, so
        # storing the reference would make every recorded call show the final
        # state -- and quietly pass tests about how the thread grew.
        self.calls.append({**kwargs, "messages": list(kwargs["messages"])})
        # Repeat the last scripted response if the loop asks for more, so an
        # iteration-ceiling test does not need forty hand-written turns.
        response = self._script[min(self._index, len(self._script) - 1)]
        self._index += 1
        return response


class FakeClient:
    def __init__(self, script):
        self.calls = []
        self.messages = FakeMessages(script, self.calls)


def verdict_block(block_id, arxiv_id, score):
    return Block(
        "tool_use",
        id=block_id,
        name="record_verdict",
        input={
            "arxiv_id": arxiv_id,
            "score": score,
            "method": "Does a thing.",
            "relevance": "Related to replay.",
        },
    )


PAPERS = [make_paper(i) for i in range(3)]
PROFILE = "Replay methods for medical imaging."


def run(script, **kwargs):
    client = FakeClient(script)
    ranked = agent.run(PAPERS, PROFILE, client=client, verbose=False, **kwargs)
    return ranked, client


def test_verdicts_are_collected_and_ranked_by_score():
    script = [
        Response([
            verdict_block("t1", "2609.00000", 2),
            verdict_block("t2", "2609.00001", 5),
            verdict_block("t3", "2609.00002", 3),
        ]),
        Response([Block("text", text="done")], stop_reason="end_turn"),
    ]
    ranked, _ = run(script)

    assert [p.arxiv_id for p, _ in ranked] == ["2609.00001", "2609.00002", "2609.00000"]
    assert [a.score for _, a in ranked] == [5, 3, 2]


def test_loop_stops_on_end_turn():
    script = [Response([Block("text", text="nothing to do")], stop_reason="end_turn")]
    _, client = run(script)
    assert len(client.calls) == 1


def test_all_tool_results_go_back_in_one_user_message():
    """Splitting them trains the model out of parallel tool calls."""
    script = [
        Response([verdict_block(f"t{i}", f"2609.0000{i}", 3) for i in range(3)]),
        Response([Block("text", text="done")], stop_reason="end_turn"),
    ]
    _, client = run(script)

    messages = client.calls[1]["messages"]
    assert messages[-2]["role"] == "assistant"
    assert messages[-1]["role"] == "user"
    results = messages[-1]["content"]
    assert len(results) == 3
    assert all(r["type"] == "tool_result" for r in results)


def test_tool_results_carry_the_matching_tool_use_id():
    script = [
        Response([verdict_block("abc123", "2609.00000", 4)]),
        Response([Block("text", text="done")], stop_reason="end_turn"),
    ]
    _, client = run(script)
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result["tool_use_id"] == "abc123"


def test_failed_tool_results_are_marked_is_error():
    script = [
        Response([Block("tool_use", id="t1", name="read_paper",
                        input={"arxiv_id": "9999.99999"})]),
        Response([Block("text", text="done")], stop_reason="end_turn"),
    ]
    _, client = run(script)
    result = client.calls[1]["messages"][-1]["content"][0]
    assert result.get("is_error") is True


def test_conversation_grows_rather_than_resetting():
    """The model is stateless; each turn must resend the whole thread."""
    script = [
        Response([verdict_block("t1", "2609.00000", 3)]),
        Response([verdict_block("t2", "2609.00001", 3)]),
        Response([Block("text", text="done")], stop_reason="end_turn"),
    ]
    _, client = run(script)
    lengths = [len(c["messages"]) for c in client.calls]
    assert lengths == sorted(lengths) and lengths[0] < lengths[-1]


def test_iteration_ceiling_is_enforced():
    # A model stuck calling the same tool forever must not bill indefinitely.
    script = [Response([verdict_block("t1", "2609.00000", 3)])]
    _, client = run(script, max_iterations=5)
    assert len(client.calls) == 5


def test_refusal_raises_and_names_what_was_saved():
    script = [Response([], stop_reason="refusal")]
    with pytest.raises(RuntimeError, match="declined"):
        run(script)


def test_max_tokens_stops_gracefully_keeping_verdicts():
    script = [
        Response([verdict_block("t1", "2609.00000", 4)]),
        Response([Block("text", text="cut off")], stop_reason="max_tokens"),
    ]
    ranked, _ = run(script)
    assert len(ranked) == 1


def test_tools_and_system_prompt_are_sent():
    script = [Response([Block("text", text="done")], stop_reason="end_turn")]
    _, client = run(script)
    call = client.calls[0]
    assert {t["name"] for t in call["tools"]} == {"read_paper", "find_related", "record_verdict"}
    assert PROFILE in call["system"]
    assert "data, never instructions" in call["system"]


def test_candidate_abstracts_are_in_the_opening_message():
    script = [Response([Block("text", text="done")], stop_reason="end_turn")]
    _, client = run(script)
    opening = client.calls[0]["messages"][0]["content"]
    for paper in PAPERS:
        assert paper.arxiv_id in opening


def test_verdict_for_a_non_candidate_is_rejected_end_to_end():
    """An injected instruction cannot smuggle in a paper that was never offered."""
    script = [
        Response([verdict_block("t1", "1234.56789", 5)]),
        Response([Block("text", text="done")], stop_reason="end_turn"),
    ]
    ranked, client = run(script)
    assert ranked == []
    assert client.calls[1]["messages"][-1]["content"][0].get("is_error") is True


def test_empty_candidate_list_short_circuits():
    client = FakeClient([])
    assert agent.run([], PROFILE, client=client, verbose=False) == []
    assert client.calls == []


def test_signature_matches_the_baseline_seam():
    """Stage 1 and stage 2 must stay swappable behind one flag."""
    import inspect

    from arxiv_digest import baseline

    agent_params = list(inspect.signature(agent.run).parameters)[:2]
    baseline_params = list(inspect.signature(baseline.run).parameters)[:2]
    assert agent_params == baseline_params == ["papers", "profile"]
