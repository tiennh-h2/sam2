"""Small CPU contract tests; run with: python -m pytest tests/test_learnable_query.py"""

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from sam2.learnable_query import SAM2LearnableQuery, instance_loss


class FakePromptEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.no_mask_embed = nn.Embedding(1, 256)

    def get_dense_pe(self):
        return torch.zeros(1, 256, 8, 8)


class FakeMaskDecoder(nn.Module):
    def forward(self, image_embeddings, sparse_prompt_embeddings, **kwargs):
        assert image_embeddings.shape[0] == sparse_prompt_embeddings.shape[0]
        mask = image_embeddings.mean(1, keepdim=True) + sparse_prompt_embeddings.mean((1, 2))[:, None, None, None]
        return mask, None, None, None


class FakeSAM(nn.Module):
    sam_prompt_embed_dim = 256
    directly_add_no_mem_embed = False
    num_feature_levels = 3

    def __init__(self):
        super().__init__()
        self.sam_prompt_encoder = FakePromptEncoder()
        self.sam_mask_decoder = FakeMaskDecoder()

    def forward_image(self, images):
        batch = images.shape[0]
        return {"backbone_fpn": [
            torch.ones(batch, 256, 32, 32),
            torch.ones(batch, 256, 16, 16),
            torch.ones(batch, 256, 8, 8),
        ]}

    def _prepare_backbone_features(self, backbone):
        levels = backbone["backbone_fpn"]
        flat = [x.flatten(2).permute(2, 0, 1) for x in levels]
        return backbone, flat, None, None


def test_queries_predict_independent_masks_and_receive_gradients():
    model = SAM2LearnableQuery(FakeSAM(), num_queries=3, chunk_size=2)
    scores, masks = model(torch.zeros(2, 3, 128, 128))
    assert scores.shape == (2, 3)
    assert masks.shape == (2, 3, 8, 8)
    (scores.sum() + masks.sum()).backward()
    assert model.head.queries.weight.grad.abs().sum() > 0


def test_empty_target_trains_queries_to_no_object():
    scores = torch.zeros(1, 3, requires_grad=True)
    masks = torch.zeros(1, 3, 8, 8, requires_grad=True)
    loss = instance_loss(scores, masks, [torch.zeros(0, 8, 8)])
    loss.backward()
    assert scores.grad is not None
    assert scores.grad.sum() > 0


def test_matched_target_supervises_a_mask():
    scores = torch.zeros(1, 3, requires_grad=True)
    masks = torch.zeros(1, 3, 8, 8, requires_grad=True)
    target = torch.zeros(1, 8, 8)
    target[0, 2:6, 2:6] = 1
    loss = instance_loss(scores, masks, [target])
    loss.backward()
    assert masks.grad.abs().sum() > 0


def test_query_count_must_cover_target_count():
    with pytest.raises(ValueError, match="more target instances"):
        instance_loss(torch.zeros(1, 2), torch.zeros(1, 2, 8, 8), [torch.zeros(3, 8, 8)])


def test_repository_loss_matches_masks_to_their_source_images():
    pytest.importorskip("tensordict")
    from training.loss_fns import MultiStepMultiMasksAndIous

    criterion = MultiStepMultiMasksAndIous(
        {"loss_mask": 1, "loss_dice": 1, "loss_iou": 0, "loss_class": 1}
    )
    scores = torch.zeros(2, 2, requires_grad=True)
    predictions = torch.zeros(2, 2, 8, 8, requires_grad=True)
    targets = torch.zeros(1, 2, 8, 8)
    targets[0, 0, :4, :4] = 1
    targets[0, 1, 4:, 4:] = 1
    outputs = [{
        "query_logits": scores,
        "query_masks": predictions,
        "video_indices": torch.tensor([0, 1]),
    }]
    loss = criterion(outputs, targets)["core_loss"]
    loss.backward()
    assert scores.grad.abs().sum() > 0
    assert predictions.grad.abs().sum() > 0


def test_sampler_can_retain_all_annotated_instances():
    from training.dataset.vos_sampler import RandomUniformSampler

    class Segments:
        def load(self, index):
            return {i: torch.ones(2, 2) for i in range(7)}

    class Frame:
        frame_idx = 0

    class Video:
        frames = [Frame()]
        video_name = "image"

    result = RandomUniformSampler(num_frames=1, max_num_objects=None).sample(
        Video(), Segments()
    )
    assert len(result.object_ids) == 7
