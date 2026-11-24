import unittest
from pathlib import Path
from integration.pipeline import run_alignment, run_fasta_pipeline


class TestGeneWeaverPipeline(unittest.TestCase):
    def test_exact_and_mismatch_alignment(self):
        results = run_alignment("chr1", "ATCGATGG", "ATCG", max_mismatches=1)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0].mismatch_count, 0)
        self.assertEqual(results[1].mismatch_count, 1)

    def test_end_to_end_fasta_to_csv(self):
        test_fasta = Path("tests/temp_test.fasta")
        test_csv = Path("tests/temp_results.csv")
        test_fasta.write_text(">chr1\nATCGATCG\n", encoding="ascii")
        try:
            results = run_fasta_pipeline(test_fasta, "ATCG", test_csv, max_mismatches=0)
            self.assertEqual(len(results), 2)
            self.assertTrue(test_csv.exists())
        finally:
            if test_fasta.exists():
                test_fasta.unlink()
            if test_csv.exists():
                test_csv.unlink()


if __name__ == "__main__":
    unittest.main()
