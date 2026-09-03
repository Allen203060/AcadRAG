import unittest
from unittest.mock import MagicMock, patch
from src.agents.arxiv_agent import score_abstracts_node

class TestArxivAgentGraph(unittest.TestCase):

    @patch("src.agents.arxiv_agent.get_llm")
    def test_score_abstracts_node(self, mock_get_llm):
        """Test that score_abstracts_node grades and ranks candidate papers correctly."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

        # Mock LLM outputs for 2 candidate papers
        resp_1 = MagicMock()
        resp_1.content = '{"score": 90, "reason": "Relevant to IoT edge face recognition."}'
        resp_2 = MagicMock()
        resp_2.content = '{"score": 30, "reason": "Irrelevant optics paper."}'
        
        mock_llm.invoke.side_effect = [resp_1, resp_2]

        state = {
            "topic": "Face Recognition on IoT Edge",
            "top_k": 1,
            "candidates": [
                {"title": "Paper A", "summary": "Abstract A", "pdf_url": "http://a.pdf", "entry_id": "1", "authors": ["Author A"]},
                {"title": "Paper B", "summary": "Abstract B", "pdf_url": "http://b.pdf", "entry_id": "2", "authors": ["Author B"]}
            ],
            "shortlist": []
        }

        output = score_abstracts_node(state)
        shortlist = output["shortlist"]

        self.assertEqual(len(shortlist), 1)
        self.assertEqual(shortlist[0]["title"], "Paper A")
        self.assertEqual(shortlist[0]["relevance_score"], 90)

if __name__ == "__main__":
    unittest.main()
