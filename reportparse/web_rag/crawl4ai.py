from sentence_transformers import SentenceTransformer, util
import torch
import asyncio
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, CacheMode
from markdown import markdown
from bs4 import BeautifulSoup
import nltk

nltk.download("punkt")
from reportparse.web_rag.search import google_search


class crawl4ai:
    def __init__(self, claim, metadata, web_sources):
        self.claim = claim
        self.meta = metadata
        self.web_sources = web_sources
        self.model = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    def get_urls(self):
        self.urls = google_search(self.claim, self.web_sources, self.meta)
        return self.urls

    def chunk_text(self, text, chunk_size=500, overlap_size=50):
        sentences = nltk.sent_tokenize(text)
        chunks = []
        current_chunk = ""
        for sentence in sentences:
            if len(current_chunk) + len(sentence) <= chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
            else:
                chunks.append(current_chunk)
                current_chunk = sentence[:overlap_size]
                current_chunk += sentence[overlap_size:]

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def get_sim_text(
        self,
        text,
        threshold=0.3,
        chunk_size=500,
    ):
        if not text:
            return []
        claim_embedding = self.model.encode(self.claim, convert_to_tensor=True)
        filtered_results = []
        chunks = self.chunk_text(text, chunk_size)
        if not chunks:
            return []
        chunk_embeddings = self.model.encode(chunks, convert_to_tensor=True)
        chunk_similarities = util.cos_sim(claim_embedding, chunk_embeddings)
        for chunk, similarity in zip(chunks, chunk_similarities[0]):
            if similarity >= threshold:
                filtered_results.append(chunk)
        return filtered_results

    def single_text_embedding(self, text):
        embedding = self.model.encode(text, convert_to_tensor=True)
        torch.cuda.empty_cache()
        return embedding

    def cos_sim(embedding1, embedding2):
        return util.pytorch_cos_sim(embedding1, embedding2).item()

    def convert_markdown_to_text(self, markdown_str: str) -> str:
        html = markdown(markdown_str)
        plain_text = BeautifulSoup(html, "html.parser").get_text()
        return plain_text

    async def run_crawler(self):
        urls = self.get_urls()
        print(urls)
        run_conf = CrawlerRunConfig(cache_mode=CacheMode.BYPASS, stream=True)

        async with AsyncWebCrawler() as crawler:
            async for result in await crawler.arun_many(urls, config=run_conf):
                if result.success:
                    print(
                        f"[OK] {result.url}, length: {len(result.markdown.raw_markdown)}"
                    )
                else:
                    print(f"[ERROR] {result.url} => {result.error_message}")

            run_conf = run_conf.clone(stream=False)
            results = await crawler.arun_many(urls, config=run_conf)
            similar_texts = []
            for res in results:
                if res.success:

                    print(f"[OK] {res.url}, length: {len(res.markdown.raw_markdown)}")
                    similar_chunks = self.get_sim_text(
                        self.convert_markdown_to_text(res.markdown.raw_markdown),
                    )
                    similar_texts.append(similar_chunks)
                else:
                    print(f"[ERROR] {res.url} => {res.error_message}")

            print("Similar texts:")
            final_info = ""
            for text in similar_texts:
                final_info += "\n".join(text) + "\n\n"
        return final_info, urls
