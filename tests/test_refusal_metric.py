import torch

from src.direction.refusal_metric import refusal_compliance_token_ids, refusal_logit_diff


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        table = {" I": [358], " Sure": [22555]}
        return table[text]


def test_refusal_compliance_token_ids_looks_up_both():
    refusal_id, compliance_id = refusal_compliance_token_ids(_FakeTokenizer())
    assert refusal_id == 358
    assert compliance_id == 22555


def test_refusal_logit_diff_is_the_right_subtraction():
    logits = torch.zeros(30000)
    logits[358] = 5.0
    logits[22555] = 2.0
    diff = refusal_logit_diff(logits, refusal_id=358, compliance_id=22555)
    assert diff.item() == 3.0


def test_refusal_logit_diff_batched():
    logits = torch.zeros(4, 30000)
    logits[:, 358] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    logits[:, 22555] = 1.0
    diff = refusal_logit_diff(logits, refusal_id=358, compliance_id=22555)
    assert torch.allclose(diff, torch.tensor([0.0, 1.0, 2.0, 3.0]))
