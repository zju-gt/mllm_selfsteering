import torch

from mllm_self_steering.vectors.extract import aggregate_vectors
from mllm_self_steering.vectors.similarity import coherence, similarity_matrix


def test_vector_forms_share_mean_norm() -> None:
    vectors = aggregate_vectors(
        {
            "a": torch.tensor([[2.0, 0.0], [2.0, 0.0]]),
            "b": torch.tensor([[0.0, 4.0], [0.0, 4.0]]),
        }
    )
    assert torch.allclose(vectors["a"]["unit"].norm(), torch.tensor(1.0))
    assert torch.allclose(vectors["b"]["unit"].norm(), torch.tensor(1.0))
    assert torch.allclose(vectors["a"]["steering"].norm(), torch.tensor(3.0))
    assert torch.allclose(vectors["b"]["steering"].norm(), torch.tensor(3.0))


def test_similarity_and_coherence() -> None:
    matrix = similarity_matrix(
        {"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
    )
    assert abs(matrix["a"]["b"]) < 1e-6
    assert coherence(torch.tensor([[1.0, 0.0], [2.0, 0.0]])) == 1.0

