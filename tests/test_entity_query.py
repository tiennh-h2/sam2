import unittest

import torch

from sam2.modeling.entity_query import EntityQueryHead, EntityQueryModel, entity_query_loss


class EntityQueryTests(unittest.TestCase):
    def test_empty_targets_train_background(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        masks = torch.zeros(1, 3, 8, 8, requires_grad=True)
        losses = entity_query_loss(
            {"pred_logits": logits, "pred_masks": masks},
            [torch.zeros(0, 8, 8)],
        )
        total = sum(losses.values())
        self.assertTrue(torch.isfinite(total))
        total.backward()
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_positive_target_trains_mask_and_objectness(self):
        logits = torch.zeros(1, 3, requires_grad=True)
        masks = torch.randn(1, 3, 8, 8, requires_grad=True)
        target = torch.zeros(1, 8, 8)
        target[:, 2:5, 2:5] = 1
        total = sum(entity_query_loss(
            {"pred_logits": logits, "pred_masks": masks}, [target]
        ).values())
        total.backward()
        self.assertGreater(masks.grad.abs().sum().item(), 0)
        self.assertGreater(logits.grad.abs().sum().item(), 0)

    def test_excess_targets_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "num_queries"):
            entity_query_loss(
                {"pred_logits": torch.zeros(1, 1), "pred_masks": torch.zeros(1, 1, 4, 4)},
                [torch.ones(2, 4, 4)],
            )

    def test_query_head_output_shapes(self):
        head = EntityQueryHead(num_queries=4, hidden_dim=32, num_heads=4)
        features = torch.randn(2, 32, 4, 4)
        pos = torch.randn_like(features)
        pixel = torch.randn(2, 32, 8, 8)
        output = head(features, pos, pixel)
        self.assertEqual(output["pred_masks"].shape, (2, 4, 8, 8))
        self.assertEqual(output["pred_logits"].shape, (2, 4))

    def test_model_uses_image_encoder_and_freezes_backbone(self):
        from torch import nn

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.neck = nn.Module()
                self.neck.d_model = 32
                self.conv = nn.Conv2d(3, 32, 1)

            def forward(self, images):
                feature = self.conv(images)
                return {"backbone_fpn": [feature, feature], "vision_pos_enc": [torch.zeros_like(feature), torch.zeros_like(feature)]}

        class SAM(nn.Module):
            def __init__(self):
                super().__init__()
                self.image_encoder = Encoder()

        model = EntityQueryModel(SAM(), num_queries=2)
        model.train()
        output = model(torch.randn(1, 3, 4, 4))
        self.assertEqual(output["pred_masks"].shape, (1, 2, 4, 4))
        self.assertFalse(model.sam_model.training)
        self.assertFalse(any(p.requires_grad for p in model.sam_model.parameters()))
