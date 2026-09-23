import unittest

from local_file_organizer.classifier import classify, OTHER_CATEGORY


class ClassifyTests(unittest.TestCase):
    def test_video_extension(self):
        self.assertEqual(classify(".mp4"), "video")
        self.assertEqual(classify(".MKV"), "video")

    def test_image_extension(self):
        self.assertEqual(classify(".jpg"), "image")
        self.assertEqual(classify("png"), "image")

    def test_document_extension(self):
        self.assertEqual(classify(".pdf"), "document")

    def test_audio_extension(self):
        self.assertEqual(classify(".flac"), "audio")

    def test_archive_extension(self):
        self.assertEqual(classify(".7z"), "archive")

    def test_software_extension(self):
        self.assertEqual(classify(".apk"), "software")

    def test_unknown_extension_is_other(self):
        self.assertEqual(classify(".xyz"), OTHER_CATEGORY)

    def test_empty_extension_is_other(self):
        self.assertEqual(classify(""), OTHER_CATEGORY)


if __name__ == "__main__":
    unittest.main()
