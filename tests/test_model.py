import unittest
from unittest.mock import patch

import torch

from src.model import (
    GroupedQueryAttention,
    MiniMistralDecoder,
    RMSNorm,
    RotaryPositionalEmbedding,
    SwiGLU,
    TokenEmbedding,
    TransformerBlock,
)


class RMSNormTests(unittest.TestCase):
    def test_normalizes_each_vector_and_preserves_shape(self) -> None:
        norm = RMSNorm(dim=2)
        values = torch.tensor([[3.0, 4.0], [0.0, 0.0]])

        result = norm(values)
        expected_first = values[0] / torch.sqrt(torch.tensor(12.5 + norm.eps))
        expected = torch.stack((expected_first, torch.zeros(2)))

        self.assertEqual(result.shape, values.shape)
        torch.testing.assert_close(result, expected)

    def test_learnable_weight_receives_gradients(self) -> None:
        norm = RMSNorm(dim=3)
        values = torch.tensor([[1.0, 2.0, 3.0]], requires_grad=True)

        norm(values).square().sum().backward()

        self.assertIsNotNone(norm.weight.grad)
        self.assertTrue(torch.isfinite(norm.weight.grad).all())
        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())

    def test_rejects_wrong_last_dimension(self) -> None:
        norm = RMSNorm(dim=3)

        with self.assertRaises(ValueError):
            norm(torch.ones(2, 4))

    def test_rejects_scalar_input(self) -> None:
        norm = RMSNorm(dim=1)

        with self.assertRaises(ValueError):
            norm(torch.tensor(1.0))


class RotaryPositionalEmbeddingTests(unittest.TestCase):
    def test_position_zero_leaves_vectors_unchanged(self) -> None:
        rope = RotaryPositionalEmbedding(head_dim=4)
        values = torch.tensor([[[[1.0, 2.0, 3.0, 4.0]]]])

        result = rope(values, torch.tensor([0]))

        torch.testing.assert_close(result, values)

    def test_rotation_preserves_each_pair_norm_and_changes_position(self) -> None:
        rope = RotaryPositionalEmbedding(head_dim=4)
        values = torch.tensor([[[[1.0, 0.0, 1.0, 0.0]]]])

        result = rope(values, torch.tensor([1]))

        self.assertFalse(torch.allclose(result, values))
        torch.testing.assert_close(
            result.square().sum(dim=-1), values.square().sum(dim=-1)
        )

    def test_accepts_nonzero_position_offsets(self) -> None:
        rope = RotaryPositionalEmbedding(head_dim=4)
        values = torch.ones(1, 2, 1, 4)

        result = rope(values, torch.tensor([12]))

        self.assertEqual(result.shape, values.shape)
        self.assertFalse(torch.allclose(result, values))

    def test_rejects_odd_head_dimension(self) -> None:
        with self.assertRaises(ValueError):
            RotaryPositionalEmbedding(head_dim=3)


class GroupedQueryAttentionTests(unittest.TestCase):
    def test_preserves_shape_with_grouped_kv_heads(self) -> None:
        attention = GroupedQueryAttention(
            dim=8, n_heads=4, n_kv_heads=2, window_size=3
        )

        result = attention(torch.randn(2, 5, 8))

        self.assertEqual(result.shape, (2, 5, 8))

    def test_does_not_attend_to_future_tokens(self) -> None:
        attention = GroupedQueryAttention(
            dim=4, n_heads=2, n_kv_heads=1, window_size=3
        )
        with torch.no_grad():
            attention.q_proj.weight.zero_()
            attention.k_proj.weight.zero_()
            attention.v_proj.weight.zero_()
            attention.v_proj.weight[0, 0] = 1.0
            attention.out_proj.weight.zero_()
            attention.out_proj.weight[0, 0] = 1.0

        first = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]]
        )
        changed_future = first.clone()
        changed_future[:, 1:, 0] = torch.tensor([30.0, 50.0])

        first_result = attention(first)
        changed_result = attention(changed_future)

        torch.testing.assert_close(first_result[:, :1], changed_result[:, :1])

    def test_sliding_window_excludes_old_tokens(self) -> None:
        attention = GroupedQueryAttention(
            dim=4, n_heads=2, n_kv_heads=1, window_size=2
        )
        with torch.no_grad():
            attention.q_proj.weight.zero_()
            attention.k_proj.weight.zero_()
            attention.v_proj.weight.zero_()
            attention.v_proj.weight[0, 0] = 1.0
            attention.out_proj.weight.zero_()
            attention.out_proj.weight[0, 0] = 1.0
        values = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0], [5.0, 0.0, 0.0, 0.0]]]
        )

        result = attention(values)

        torch.testing.assert_close(result[0, 2, 0], torch.tensor(4.0))

    def test_rejects_incompatible_head_counts(self) -> None:
        with self.assertRaises(ValueError):
            GroupedQueryAttention(
                dim=8, n_heads=4, n_kv_heads=3, window_size=2
            )

    def test_uses_absolute_positions_for_sliding_window(self) -> None:
        attention = GroupedQueryAttention(
            dim=4, n_heads=2, n_kv_heads=1, window_size=2
        )
        with torch.no_grad():
            attention.q_proj.weight.zero_()
            attention.k_proj.weight.zero_()
            attention.v_proj.weight.zero_()
            attention.v_proj.weight[0, 0] = 1.0
            attention.out_proj.weight.zero_()
            attention.out_proj.weight[0, 0] = 1.0
        values = torch.tensor(
            [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]]
        )

        result = attention(values, position_ids=torch.tensor([10, 11]))

        torch.testing.assert_close(result[0, 1, 0], torch.tensor(2.0))


class SwiGLUTests(unittest.TestCase):
    def test_preserves_batch_and_sequence_dimensions(self) -> None:
        network = SwiGLU(dim=8, hidden_dim=16)

        result = network(torch.randn(2, 5, 8))

        self.assertEqual(result.shape, (2, 5, 8))

    def test_parameters_receive_gradients(self) -> None:
        network = SwiGLU(dim=4, hidden_dim=8)
        values = torch.randn(2, 3, 4, requires_grad=True)

        network(values).square().sum().backward()

        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        for parameter in network.parameters():
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())


class TransformerBlockTests(unittest.TestCase):
    def test_preserves_shape_and_accepts_absolute_positions(self) -> None:
        block = TransformerBlock(
            dim=8,
            n_heads=4,
            n_kv_heads=2,
            window_size=3,
            hidden_dim=16,
        )

        result = block(torch.randn(2, 4, 8), position_ids=torch.arange(10, 14))

        self.assertEqual(result.shape, (2, 4, 8))

    def test_gradients_flow_through_both_residual_branches(self) -> None:
        block = TransformerBlock(
            dim=8,
            n_heads=4,
            n_kv_heads=2,
            window_size=3,
            hidden_dim=16,
        )
        values = torch.randn(1, 3, 8, requires_grad=True)

        block(values).square().sum().backward()

        self.assertIsNotNone(values.grad)
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertIsNotNone(block.attention.q_proj.weight.grad)
        self.assertIsNotNone(block.feed_forward.gate_proj.weight.grad)

    def test_rejects_non_sequence_input(self) -> None:
        block = TransformerBlock(
            dim=8,
            n_heads=4,
            n_kv_heads=2,
            window_size=3,
            hidden_dim=16,
        )

        with self.assertRaises(ValueError):
            block(torch.randn(2, 8))


class TokenEmbeddingTests(unittest.TestCase):
    def test_maps_token_ids_to_their_learned_vectors(self) -> None:
        embedding = TokenEmbedding(vocab_size=4, dim=3)
        expected = torch.tensor(
            [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6],
             [0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
        )
        with torch.no_grad():
            embedding.embedding.weight.copy_(expected)

        result = embedding(torch.tensor([[2, 0], [3, 1]]))

        expected_result = torch.stack((expected[[2, 0]], expected[[3, 1]]))
        torch.testing.assert_close(result, expected_result)

    def test_preserves_batch_and_sequence_shape(self) -> None:
        embedding = TokenEmbedding(vocab_size=20, dim=8)

        result = embedding(torch.randint(0, 20, (2, 5)))

        self.assertEqual(result.shape, (2, 5, 8))

    def test_supports_int32_token_ids(self) -> None:
        embedding = TokenEmbedding(vocab_size=5, dim=4)

        result = embedding(torch.tensor([[1, 3]], dtype=torch.int32))

        self.assertEqual(result.shape, (1, 2, 4))

    def test_rejects_non_integer_token_ids(self) -> None:
        embedding = TokenEmbedding(vocab_size=5, dim=4)

        with self.assertRaises(TypeError):
            embedding(torch.tensor([[1.0, 2.0]]))

    def test_rejects_input_without_batch_and_sequence_dimensions(self) -> None:
        embedding = TokenEmbedding(vocab_size=5, dim=4)

        with self.assertRaises(ValueError):
            embedding(torch.tensor([1, 2]))

    def test_embedding_weights_receive_gradients(self) -> None:
        embedding = TokenEmbedding(vocab_size=5, dim=4)

        embedding(torch.tensor([[1, 3]])).square().sum().backward()

        self.assertIsNotNone(embedding.embedding.weight.grad)
        self.assertTrue(torch.isfinite(embedding.embedding.weight.grad).all())


class MiniMistralDecoderTests(unittest.TestCase):
    def create_model(self) -> MiniMistralDecoder:
        return MiniMistralDecoder(
            vocab_size=32,
            dim=16,
            n_layers=2,
            n_heads=4,
            n_kv_heads=2,
            window_size=4,
            hidden_dim=32,
        )

    def test_maps_batch_of_token_sequences_to_logits(self) -> None:
        model = self.create_model()

        logits = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(logits.shape, (2, 3, 32))

    def test_accepts_absolute_positions_for_prefill_chunks(self) -> None:
        model = self.create_model()

        logits = model(torch.tensor([[1, 2, 3]]), position_ids=torch.arange(10, 13))

        self.assertEqual(logits.shape, (1, 3, 32))

    def test_gradients_flow_from_logits_to_embeddings_and_layers(self) -> None:
        model = self.create_model()

        model(torch.tensor([[1, 2, 3]])).square().mean().backward()

        self.assertIsNotNone(model.token_embedding.embedding.weight.grad)
        self.assertIsNotNone(model.layers[0].attention.q_proj.weight.grad)
        self.assertIsNotNone(model.layers[1].feed_forward.gate_proj.weight.grad)
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_requires_at_least_one_layer(self) -> None:
        with self.assertRaises(ValueError):
            MiniMistralDecoder(
                vocab_size=32,
                dim=16,
                n_layers=0,
                n_heads=4,
                n_kv_heads=2,
                window_size=4,
                hidden_dim=32,
            )

    def test_rejects_empty_token_sequence(self) -> None:
        model = self.create_model()

        with self.assertRaises(ValueError):
            model(torch.empty((1, 0), dtype=torch.long))

    def test_greedy_generation_appends_requested_number_of_tokens(self) -> None:
        model = self.create_model()
        with torch.no_grad():
            model.lm_head.weight.zero_()
        prompt = torch.tensor([[4, 7]])

        generated = model.generate(prompt, max_new_tokens=3)

        self.assertEqual(generated.shape, (1, 5))
        torch.testing.assert_close(generated[:, :2], prompt)
        torch.testing.assert_close(generated[:, 2:], torch.zeros((1, 3), dtype=torch.long))

    def test_stops_when_eos_token_is_generated(self) -> None:
        model = self.create_model()
        with torch.no_grad():
            model.lm_head.weight.zero_()

        generated = model.generate(
            torch.tensor([[4, 7]]), max_new_tokens=5, eos_token_id=0
        )

        self.assertEqual(generated.shape, (1, 3))
        self.assertEqual(generated[0, -1].item(), 0)

    def test_finished_rows_keep_eos_while_other_batch_rows_continue(self) -> None:
        model = self.create_model()

        def predictable_logits(token_ids: torch.Tensor, **_: torch.Tensor) -> torch.Tensor:
            logits = torch.zeros(token_ids.shape[0], token_ids.shape[1], 32)
            for row, prompt_id in enumerate(token_ids[:, 0].tolist()):
                if prompt_id == 1 or token_ids.shape[1] == 3:
                    logits[row, -1, 0] = 1.0
                else:
                    logits[row, -1, 3] = 1.0
            return logits

        with patch.object(model, "forward", side_effect=predictable_logits):
            generated = model.generate(
                torch.tensor([[1], [2]]), max_new_tokens=5, eos_token_id=0
            )

        self.assertEqual(generated.shape, (2, 4))
        torch.testing.assert_close(
            generated[0], torch.tensor([1, 0, 0, 0], dtype=torch.long)
        )
        torch.testing.assert_close(
            generated[1], torch.tensor([2, 3, 3, 0], dtype=torch.long)
        )

    def test_zero_new_tokens_returns_prompt_unchanged(self) -> None:
        model = self.create_model()
        prompt = torch.tensor([[4, 7]])

        generated = model.generate(prompt, max_new_tokens=0)

        torch.testing.assert_close(generated, prompt)
        self.assertIsNot(generated, prompt)

    def test_rejects_negative_generation_length(self) -> None:
        model = self.create_model()

        with self.assertRaises(ValueError):
            model.generate(torch.tensor([[1]]), max_new_tokens=-1)


if __name__ == "__main__":
    unittest.main()
