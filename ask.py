"""Ask a question from the command line: python ask.py "What is a Loan Estimate?" """
import sys
from pathlib import Path

from dotenv import load_dotenv

from rag.generator import GeminiGenerator, answer
from rag.retriever import SentenceTransformerEmbedder, VectorIndex

if __name__ == "__main__":
    load_dotenv()
    question = " ".join(sys.argv[1:]) or input("Question: ")
    index = VectorIndex.load(Path("index"), SentenceTransformerEmbedder())
    result = answer(question, index, GeminiGenerator())
    print("\n" + result["answer"] + "\n")
    for i, s in enumerate(result["sources"], 1):
        print(f"[{i}] {s['source']} p.{s['page']} (similarity {s['score']})")
