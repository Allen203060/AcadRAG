import unittest
from unittest.mock import MagicMock, patch
from src.agents.arxiv_agent import score_abstracts_node, download_ingest_node

class TestArxivAgentGraph(unittest.TestCase):

    @patch("src.agents.arxiv_agent.get_llm")
    def test_score_abstracts_node(self, mock_get_llm):
        """Test that score_abstracts_node grades and ranks candidate papers correctly."""
        mock_llm = MagicMock()
        mock_get_llm.return_value = mock_llm

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

    @patch("src.agents.arxiv_agent.populate_databases")
    @patch("src.agents.arxiv_agent.extract_pdf_with_docling")
    @patch("src.agents.arxiv_agent.requests.get")
    def test_download_ingest_node(self, mock_requests_get, mock_docling, mock_populate):
        """Test that download_ingest_node correctly downloads PDFs via HTTP and calls Docling DOM parsing."""
        mock_response = MagicMock()
        mock_response.content = b"%PDF-1.4 Mock PDF Content"
        mock_requests_get.return_value = mock_response

        state = {
            "shortlist": [
                {"title": "Test Paper", "pdf_url": "http://example.com/test.pdf"}
            ]
        }

        output = download_ingest_node(state)

        mock_requests_get.assert_called_once_with("http://example.com/test.pdf", timeout=30)
        mock_docling.assert_called_once()
        mock_populate.assert_called_once()
        self.assertEqual(output, {})

if __name__ == "__main__":
    unittest.main()
