import pytest
import torch

from models.short_horizon_gru import ShortHorizonTailGRU


def test_short_horizon_output_shapes() -> None:
    model = ShortHorizonTailGRU(
        history_steps=20,
        future_steps=10,
        hidden_size=32,
        num_layers=1,
    )
    past = torch.randn(4, 50, 2)

    future, tail_logits = model(past)

    assert future.shape == (4, 10, 2)
    assert tail_logits.shape == (4,)


def test_prediction_starts_from_last_observation() -> None:
    model = ShortHorizonTailGRU(
        history_steps=5,
        future_steps=3,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
    )
    for parameter in model.parameters():
        torch.nn.init.zeros_(parameter)

    past = torch.tensor(
        [[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]]]
    )
    future, _ = model(past)

    expected = past[:, -1:, :].expand(-1, 3, -1)
    assert torch.allclose(future, expected)


def test_rejects_insufficient_history() -> None:
    model = ShortHorizonTailGRU(history_steps=20, future_steps=10)

    with pytest.raises(ValueError, match="at least 20"):
        model(torch.randn(2, 19, 2))
