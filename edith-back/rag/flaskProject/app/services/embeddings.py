# embeddings.py
from langchain_chroma import Chroma
from app.services.codebert_model import get_code_embedding
from langchain.embeddings.base import Embeddings

# 래퍼 클래스 생성
class GraphCodeBERTEmbeddings(Embeddings):
    def embed_documents(self, texts):
        return [self.embed_query(text) for text in texts]

    def embed_query(self, code_snippet):
        return get_code_embedding(code_snippet)

class CodeEmbeddingProcessor:
    def __init__(self, uuid):
        self.db = Chroma(
            embedding_function=GraphCodeBERTEmbeddings(),
            collection_name=f'code_embeddings_{uuid}',
            persist_directory=None
        )

    # Chunk 코드 임베딩
    def store_embeddings(self, code_snippets):
        try:
            self.db.add_texts(
                texts=code_snippets,
            )
            return True
        except Exception as e:
            print(f"Error storing embeddings: {e}")
            return False

    # 유사 코드 검색
    def query_similar_code(self, code_snippet, n_results=5):
        try:
            results = self.db.similarity_search_with_score(
                query=code_snippet,
                k=n_results
            )
            related_codes = [doc.page_content for doc, score in results]
            return related_codes
        except Exception as e:
            print(f"Error querying similar code: {e}")
            return []

    def cleanup(self):
        try:
            # 시스템 캐시 정리
            if hasattr(self.db, '_client'):
                self.db._client.clear_system_cache()

            # Collection 정리
            if hasattr(self.db, '_collection'):
                try:
                    # 모든 데이터 삭제
                    self.db._collection.delete(where={})
                    # Collection 자체 삭제
                    self.db._collection.delete()
                except Exception as collection_error:
                    print(f"Warning during collection cleanup: {collection_error}")

            # Collection 삭제
            try:
                self.db.delete_collection()
            except Exception as delete_error:
                print(f"Warning during delete_collection: {delete_error}")

            # 메모리에서 객체 정리
            self.db = None

            # 가비지 컬렉션 강제 실행 (선택적)
            import gc
            gc.collect()

            return True
        except Exception as e:
            print(f"Critical error during cleanup: {e}")
            return False