import unittest
import torch

from wan22_opsd_bounded_causal_attention import BoundedLayerKVCache
from wan22_causal_attention import LayerKVCache
from wan22_opsd_prefix_cache import (
    append_cached_chunk_, clone_prefix, extract_latest_chunk,
    replace_chunk_from_replay, truncate_full_cache_, update_ema_,
)


def cache(values):
    tensor = torch.tensor(values, dtype=torch.float32).view(1, -1, 1)
    positions = torch.arange(tensor.shape[1])
    return BoundedLayerKVCache(tensor.clone(), (tensor + 100).clone(), positions, tensor.shape[1])


class OPSDPrefixCacheTest(unittest.TestCase):
    def test_delayed_replacement_preserves_latest_generated_chunk(self):
        teacher = [cache([10, 11, 20, 21, 30, 31])]
        replay = clone_prefix(teacher, 2)
        replay[0].key = torch.cat((replay[0].key, torch.tensor([[[120.0], [121.0]]])), dim=1)
        replay[0].value = torch.cat((replay[0].value, torch.tensor([[[220.0], [221.0]]])), dim=1)
        replay[0].positions = torch.arange(4)
        replay[0].seen_tokens = 4

        replace_chunk_from_replay(teacher, replay, start_token=2, chunk_tokens=2)

        self.assertEqual(teacher[0].key.flatten().tolist(), [10, 11, 120, 121, 30, 31])
        self.assertEqual(teacher[0].value.flatten().tolist(), [110, 111, 220, 221, 130, 131])
        self.assertEqual(teacher[0].seen_tokens, 6)

    def test_clone_prefix_is_detached_and_independent(self):
        original = [cache([1, 2, 3, 4])]
        cloned = clone_prefix(original, 2)
        cloned[0].key.add_(10)
        self.assertEqual(original[0].key.flatten().tolist(), [1, 2, 3, 4])
        self.assertEqual(cloned[0].seen_tokens, 2)

    def test_replacement_rejects_wrong_replay_endpoint(self):
        with self.assertRaisesRegex(ValueError, "end exactly"):
            replace_chunk_from_replay([cache([1, 2, 3, 4])], [cache([8, 9])], start_token=0, chunk_tokens=1)

    def test_ema_update(self):
        parameter = torch.nn.Parameter(torch.tensor([4.0]))
        ema = {"weight": torch.tensor([0.0])}
        update_ema_(ema, [("weight", parameter)], 0.75)
        self.assertAlmostEqual(ema["weight"].item(), 1.0)

    def test_full_cache_in_place_replay_preserves_latest_generated_chunk(self):
        values = torch.tensor([10, 11, 20, 21, 30, 31], dtype=torch.float32).view(1, -1, 1)
        teacher = [LayerKVCache(values.clone(), (values + 100).clone(), 6)]
        latest = extract_latest_chunk(teacher, 2)
        truncate_full_cache_(teacher, 2)
        clean = torch.tensor([120, 121], dtype=torch.float32).view(1, -1, 1)
        teacher[0].key = torch.cat((teacher[0].key, clean), dim=1)
        teacher[0].value = torch.cat((teacher[0].value, clean + 100), dim=1)
        teacher[0].seen_tokens = 4
        append_cached_chunk_(teacher, latest, expected_start=4)
        self.assertEqual(teacher[0].key.flatten().tolist(), [10, 11, 120, 121, 30, 31])
        self.assertEqual(teacher[0].seen_tokens, 6)


if __name__ == "__main__":
    unittest.main()
