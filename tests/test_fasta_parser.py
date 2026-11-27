import unittest
from pathlib import Path
from parser.fasta_parser import parse_fasta


class TestFastaParser(unittest.TestCase):

    def test_valid_fasta_file(self):
        test_file = Path("tests/temp_valid.fasta")
        test_file.write_text(">chr1\natcg\n", encoding="ascii")

        try:
            records = parse_fasta(test_file)
            self.assertEqual(records, [("chr1", "ATCG")])
        finally:
            test_file.unlink(missing_ok=True)

    def test_missing_fasta_file(self):
        with self.assertRaises(FileNotFoundError):
            parse_fasta("tests/missing.fasta")

    def test_empty_fasta_file(self):
        test_file = Path("tests/temp_empty.fasta")
        test_file.write_text("", encoding="ascii")

        try:
            records = parse_fasta(test_file)
            self.assertEqual(records, [])
        finally:
            test_file.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()