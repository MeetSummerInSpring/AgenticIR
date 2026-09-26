from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from demo.replay.builder import parse_qa_log, parse_workflow_log


class ReplayParserTest(unittest.TestCase):
    def test_parses_workflow_records(self) -> None:
        content = """\
2026-01-01 10:00:00,000 - INFO
Evaluation: [('rain', 'high')]

2026-01-01 10:00:02,500 - INFO
Plan: ['deraining']

"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workflow.log"
            path.write_text(content, encoding="utf-8")
            records = parse_workflow_log(path)

        self.assertEqual(len(records), 2)
        self.assertEqual(records[0]["message"], "Evaluation: [('rain', 'high')]")
        self.assertEqual(records[1]["timestamp"].microsecond, 500000)

    def test_parses_qa_without_base64(self) -> None:
        content = """\
_Note: test

**Question**

What's the severity?

![image](data:image/png;base64,SECRET)

**Answer (from DepictQA)**

[('rain', 'very high')]
"""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "llm_qa.md"
            path.write_text(content, encoding="utf-8")
            interactions = parse_qa_log(path)

        self.assertEqual(len(interactions), 1)
        self.assertEqual(interactions[0]["source"], "depictqa")
        self.assertNotIn("SECRET", interactions[0]["question"])
        self.assertEqual(interactions[0]["parsed"], [("rain", "very high")])


if __name__ == "__main__":
    unittest.main()
