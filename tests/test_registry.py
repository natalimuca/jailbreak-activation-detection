"""Pure-data sanity checks for the SAE provider registry -- no network/model
needed, so no marker (see tests/test_llama_scope.py for the network-marked
convention this deliberately doesn't need)."""

from src.sae.registry import SAE_PROVIDERS


def test_every_provider_has_a_callable_loader():
    for model_name, (load_sae, layers, micro_batch_size) in SAE_PROVIDERS.items():
        assert callable(load_sae), model_name


def test_every_provider_has_nonempty_layers():
    for model_name, (load_sae, layers, micro_batch_size) in SAE_PROVIDERS.items():
        assert len(layers) > 0, model_name
        assert all(isinstance(l, int) for l in layers), model_name


def test_micro_batch_size_is_none_or_positive_int():
    for model_name, (load_sae, layers, micro_batch_size) in SAE_PROVIDERS.items():
        assert micro_batch_size is None or (isinstance(micro_batch_size, int) and micro_batch_size > 0), model_name
