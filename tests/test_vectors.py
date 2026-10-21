import torch

from mllm_self_steering.utils.io import sha256_file
from mllm_self_steering.vectors.extract import aggregate_vectors
from mllm_self_steering.vectors.similarity import coherence, similarity_matrix
from mllm_self_steering.vectors.storage import export_vllm_hook


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


def test_vllm_exports_are_byte_stable(tmp_path) -> None:
    vectors = {
        "capability": {
            "raw": torch.tensor([1.0, 2.0]),
            "unit": torch.tensor([0.5, 1.0]),
            "steering": torch.tensor([2.0, 4.0]),
        }
    }
    export_vllm_hook(tmp_path, vectors)
    destination = tmp_path / "capability__steering.pt"
    first = sha256_file(destination)

    export_vllm_hook(tmp_path, vectors)

    assert sha256_file(destination) == first

