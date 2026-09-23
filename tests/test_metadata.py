import unittest

from local_file_organizer.metadata import human_readable_size


class HumanReadableSizeTests(unittest.TestCase):
    def test_bytes(self):
        self.assertEqual(human_readable_size(500), "500 B")

    def test_kilobytes(self):
        self.assertEqual(human_readable_size(2048), "2.00 KB")

    def test_megabytes(self):
        self.assertEqual(human_readable_size(5 * 1024 * 1024), "5.00 MB")

    def test_gigabytes(self):
        self.assertEqual(human_readable_size(3 * 1024 ** 3), "3.00 GB")

    def test_zero(self):
        self.assertEqual(human_readable_size(0), "0 B")

    def test_negative_raises(self):
        with self.assertRaises(ValueError):
            human_readable_size(-1)


if __name__ == "__main__":
    unittest.main()
