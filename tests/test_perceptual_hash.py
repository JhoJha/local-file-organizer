import tempfile
import unittest
from pathlib import Path
from unittest import mock

from PIL import Image

from local_file_organizer.perceptual_hash import (
    DEFAULT_SIMILARITY_THRESHOLD,
    SimilarityRecord,
    compute_dhash,
    get_image_metadata,
    group_similar_images,
    hamming_distance,
)


def _save_image(path: Path, color, size=(64, 64), fmt="PNG"):
    Image.new("RGB", size, color=color).save(path, format=fmt)


def _save_gradient(path: Path, size=(64, 64), fmt="PNG"):
    img = Image.new("RGB", size)
    pixels = img.load()
    for x in range(size[0]):
        for y in range(size[1]):
            value = int(255 * x / size[0])
            pixels[x, y] = (value, value, value)
    img.save(path, format=fmt)


class ComputeDhashTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dir = Path(self.tmpdir.name)

    def test_hash_is_16_hex_chars_for_default_size(self):
        path = self.dir / "solid.png"
        _save_image(path, (200, 50, 50))
        digest = compute_dhash(path)
        self.assertEqual(len(digest), 16)
        int(digest, 16)  # must be valid hex

    def test_identical_images_have_zero_distance(self):
        path_a = self.dir / "a.png"
        path_b = self.dir / "b.png"
        _save_gradient(path_a)
        _save_gradient(path_b)
        self.assertEqual(
            hamming_distance(compute_dhash(path_a), compute_dhash(path_b)), 0
        )

    def test_slightly_resized_copy_has_low_distance(self):
        original = self.dir / "orig.png"
        resized = self.dir / "resized.png"
        _save_gradient(original, size=(200, 200))
        img = Image.open(original).resize((180, 180), Image.Resampling.LANCZOS)
        img.save(resized, format="PNG")

        distance = hamming_distance(compute_dhash(original), compute_dhash(resized))
        self.assertLessEqual(distance, DEFAULT_SIMILARITY_THRESHOLD)

    def test_very_different_images_have_high_distance(self):
        black = self.dir / "black.png"
        white = self.dir / "white.png"
        _save_image(black, (0, 0, 0))
        _save_image(white, (255, 255, 255))
        # A solid image dHash is all-zero regardless of the color, since
        # every adjacent-pixel comparison is equal (not strictly greater).
        # Use a checkerboard vs. a gradient instead to get real contrast.
        checker = self.dir / "checker.png"
        img = Image.new("RGB", (64, 64))
        pixels = img.load()
        for x in range(64):
            for y in range(64):
                pixels[x, y] = (255, 255, 255) if (x // 8 + y // 8) % 2 == 0 else (0, 0, 0)
        img.save(checker, format="PNG")

        gradient = self.dir / "gradient.png"
        _save_gradient(gradient)

        distance = hamming_distance(compute_dhash(checker), compute_dhash(gradient))
        self.assertGreater(distance, DEFAULT_SIMILARITY_THRESHOLD)

    def test_exact_duplicate_sha256_is_a_different_concept_than_similarity(self):
        # Two images with different exact bytes (different content, so no
        # SHA-256 duplicate) can still be perceptually near-identical.
        path_a = self.dir / "grad_png.png"
        path_b = self.dir / "grad_jpg.jpg"
        _save_gradient(path_a, fmt="PNG")
        _save_gradient(path_b, fmt="JPEG")

        self.assertNotEqual(path_a.read_bytes(), path_b.read_bytes())
        distance = hamming_distance(compute_dhash(path_a), compute_dhash(path_b))
        self.assertLessEqual(distance, DEFAULT_SIMILARITY_THRESHOLD)

    def test_corrupt_image_raises(self):
        fake = self.dir / "not-an-image.png"
        fake.write_bytes(b"this is not a valid image file")
        with self.assertRaises(Exception):
            compute_dhash(fake)

    def test_dhash_determinism_with_flattened_data_and_legacy_fallback(self):
        # Verify that get_flattened_data and legacy getdata produce identical dhash
        path = self.dir / "sample.png"
        _save_gradient(path)
        hash_standard = compute_dhash(path)

        # Mock Image.Image to remove get_flattened_data to test legacy fallback
        with mock.patch.object(Image.Image, "get_flattened_data", create=True) as mocked:
            del Image.Image.get_flattened_data
            try:
                hash_legacy = compute_dhash(path)
            finally:
                Image.Image.get_flattened_data = mocked

        self.assertEqual(hash_standard, hash_legacy)
        self.assertEqual(len(hash_standard), 16)


class GetImageMetadataTests(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmpdir.cleanup)
        self.dir = Path(self.tmpdir.name)

    def test_valid_image_returns_hash_and_method(self):
        path = self.dir / "ok.png"
        _save_gradient(path)
        result = get_image_metadata(path)
        self.assertIsNone(result.error)
        self.assertIsNotNone(result.perceptual_hash)
        self.assertEqual(result.similarity_method, "dhash8x8")

    def test_corrupt_image_returns_error_not_raise(self):
        path = self.dir / "bad.png"
        path.write_bytes(b"garbage-not-a-png")
        result = get_image_metadata(path)
        self.assertIsNone(result.perceptual_hash)
        self.assertIsNotNone(result.error)


class GroupSimilarImagesTests(unittest.TestCase):
    def test_groups_within_threshold_and_leaves_singletons_ungrouped(self):
        records = [
            SimilarityRecord(path="/a.png", perceptual_hash="0" * 16),
            SimilarityRecord(path="/b.png", perceptual_hash="0" * 16),
            SimilarityRecord(path="/c.png", perceptual_hash="f" * 16),
        ]
        stats = group_similar_images(records, threshold=8)
        self.assertEqual(stats.group_by_path["/a.png"], "SIM_00001")
        self.assertEqual(stats.group_by_path["/b.png"], "SIM_00001")
        self.assertNotIn("/c.png", stats.group_by_path)
        self.assertEqual(stats.total_groups, 1)
        self.assertEqual(stats.total_grouped_files, 2)
        self.assertEqual(stats.group_max_distance["SIM_00001"], 0)

    def test_group_ids_are_sequential_and_deterministic(self):
        records = [
            SimilarityRecord(path="/z1.png", perceptual_hash="0" * 16),
            SimilarityRecord(path="/z2.png", perceptual_hash="0" * 16),
            SimilarityRecord(path="/a1.png", perceptual_hash="f" * 16),
            SimilarityRecord(path="/a2.png", perceptual_hash="f" * 16),
        ]
        stats = group_similar_images(records, threshold=0)
        # Ordered by smallest path within each cluster: "/a1.png" < "/z1.png"
        self.assertEqual(stats.group_by_path["/a1.png"], "SIM_00001")
        self.assertEqual(stats.group_by_path["/a2.png"], "SIM_00001")
        self.assertEqual(stats.group_by_path["/z1.png"], "SIM_00002")
        self.assertEqual(stats.group_by_path["/z2.png"], "SIM_00002")

    def test_no_records_yields_empty_stats(self):
        stats = group_similar_images([])
        self.assertEqual(stats.group_by_path, {})
        self.assertEqual(stats.total_groups, 0)
        self.assertEqual(stats.total_grouped_files, 0)


class ConservativeClusteringV211Tests(unittest.TestCase):
    """V2.1.1: complete-linkage — max_pairwise_distance <= threshold."""

    # A-B=6, B-C=7, A-C=13 (verified by hand: 0x3F=6 bits,
    # 0x1FFF=13 bits, 0x1FC0=7 bits).
    HASH_A = "0000000000000000"
    HASH_B = "000000000000003f"
    HASH_C = "0000000000001fff"

    def _transitive_records(self):
        return [
            SimilarityRecord(path="/a.png", perceptual_hash=self.HASH_A),
            SimilarityRecord(path="/b.png", perceptual_hash=self.HASH_B),
            SimilarityRecord(path="/c.png", perceptual_hash=self.HASH_C),
        ]

    def test_transitive_chain_does_not_merge_all_three(self):
        self.assertEqual(hamming_distance(self.HASH_A, self.HASH_B), 6)
        self.assertEqual(hamming_distance(self.HASH_B, self.HASH_C), 7)
        self.assertEqual(hamming_distance(self.HASH_A, self.HASH_C), 13)

        stats = group_similar_images(self._transitive_records(), threshold=8)
        # A and B stay together (6 <= 8); C cannot join because 13 > 8.
        self.assertIn("/a.png", stats.group_by_path)
        self.assertEqual(
            stats.group_by_path["/a.png"], stats.group_by_path["/b.png"]
        )
        self.assertNotEqual(
            stats.group_by_path.get("/c.png"),
            stats.group_by_path["/a.png"],
        )
        # C is a singleton here, so it stays ungrouped.
        self.assertNotIn("/c.png", stats.group_by_path)
        self.assertEqual(stats.group_max_distance["SIM_00001"], 6)

    def test_max_pairwise_invariant_holds_for_every_group(self):
        records = self._transitive_records() + [
            SimilarityRecord(path="/d.png", perceptual_hash="0" * 16),
            SimilarityRecord(path="/e.png", perceptual_hash="0" * 15 + "3"),
            SimilarityRecord(path="/f.png", perceptual_hash="f" * 16),
        ]
        threshold = 8
        stats = group_similar_images(records, threshold=threshold)
        by_group: dict[str, list[str]] = {}
        path_to_hash = {r.path: r.perceptual_hash for r in records}
        for path, group_id in stats.group_by_path.items():
            by_group.setdefault(group_id, []).append(path_to_hash[path])
        self.assertGreaterEqual(len(by_group), 1)
        for group_id, hashes in by_group.items():
            observed = stats.group_max_distance[group_id]
            recomputed = 0
            for i in range(len(hashes)):
                for j in range(i + 1, len(hashes)):
                    recomputed = max(
                        recomputed, hamming_distance(hashes[i], hashes[j])
                    )
            self.assertEqual(observed, recomputed)
            self.assertLessEqual(observed, threshold)

    def test_valid_clique_stays_together(self):
        # A=0 bits, B=2 bits, C=3 bits nested: all pairs <= 8.
        records = [
            SimilarityRecord(path="/a.png", perceptual_hash="0" * 16),
            SimilarityRecord(path="/b.png", perceptual_hash="0" * 15 + "3"),
            SimilarityRecord(path="/c.png", perceptual_hash="0" * 15 + "7"),
        ]
        stats = group_similar_images(records, threshold=8)
        self.assertEqual(stats.total_groups, 1)
        self.assertEqual(stats.total_grouped_files, 3)
        group_ids = set(stats.group_by_path.values())
        self.assertEqual(group_ids, {"SIM_00001"})
        self.assertLessEqual(stats.group_max_distance["SIM_00001"], 8)

    def test_determinism_same_input_same_output(self):
        records = self._transitive_records()
        first = group_similar_images(records, threshold=8)
        second = group_similar_images(records, threshold=8)
        self.assertEqual(first.group_by_path, second.group_by_path)
        self.assertEqual(first.group_max_distance, second.group_max_distance)

    def test_input_order_is_normalized(self):
        ordered = self._transitive_records()
        reversed_records = list(reversed(ordered))
        shuffled = [ordered[1], ordered[2], ordered[0]]
        expected = group_similar_images(ordered, threshold=8)
        for variant in (reversed_records, shuffled):
            stats = group_similar_images(variant, threshold=8)
            self.assertEqual(stats.group_by_path, expected.group_by_path)
            self.assertEqual(
                stats.group_max_distance, expected.group_max_distance
            )

    def test_threshold_is_configurable(self):
        records = self._transitive_records()
        loose = group_similar_images(records, threshold=13)
        strict = group_similar_images(records, threshold=8)
        tighter = group_similar_images(records, threshold=2)
        # threshold=13 admits the full clique (max 13); 8 splits C off;
        # 2 splits everything (A-B=6 > 2).
        self.assertEqual(loose.total_grouped_files, 3)
        self.assertEqual(loose.group_max_distance["SIM_00001"], 13)
        self.assertEqual(strict.total_grouped_files, 2)
        self.assertEqual(tighter.total_groups, 0)
        self.assertEqual(tighter.group_by_path, {})


if __name__ == "__main__":
    unittest.main()
