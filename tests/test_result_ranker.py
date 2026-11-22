# tests/test_result_ranker.py
from src.result_ranker import rank_guides

def test_rank_guides():
    test_data = [
        {"id": "g1", "score": 50},
        {"id": "g2", "score": 99},
        {"id": "g3", "score": 75}
    ]
    ranked = rank_guides(test_data, top_n=2)
    assert len(ranked) == 2
    assert ranked[0]["id"] == "g2" # Highest score first
    assert ranked[1]["id"] == "g3" # Second highest