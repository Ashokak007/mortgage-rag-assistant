"""Read every PDF in data/, chunk it, embed it, and save the index to index/."""
import argparse
import time
from pathlib import Path

from rag.ingest import load_folder
from rag.retriever import SentenceTransformerEmbedder, VectorIndex

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--out", default="index")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--overlap", type=int, default=150)
    args = ap.parse_args()

    t0 = time.time()
    chunks = load_folder(Path(args.data), args.chunk_size, args.overlap)
    pages = len({(c.source, c.page) for c in chunks})
    docs = len({c.source for c in chunks})
    print(f"Loaded {docs} PDFs, {pages} pages -> {len(chunks)} chunks")
    index = VectorIndex(SentenceTransformerEmbedder()).build(chunks)
    index.save(Path(args.out))
    print(f"Saved index to {args.out}/ in {time.time() - t0:.1f}s")
