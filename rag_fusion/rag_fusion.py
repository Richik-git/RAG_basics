import logging
import os
from langchain_core.documents import Document
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_core.retrievers import BaseRetriever
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field


_log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rag_fusion.log")

logger = logging.getLogger("rag_fusion_logger")
logger.setLevel(logging.INFO)
logger.propagate = False

if not logger.handlers:
    _file_handler = logging.FileHandler(_log_path)
    _file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(_file_handler)
    
class SubQuerySchema(BaseModel):
    
    sub_queries: list[str] = Field(..., description="List of sub-queries to be generated from the main query.")



class RAGFusion:
    '''
    It combines multiple retrievers or Sub-queries.
    It uses an ensemble of retrievers to retrieve docs and 
    then fuses the result to give the final response
    '''
    
    def __init__(self,
                retriever: BaseRetriever,
                llm_chain: Runnable | None = None,
                num_subqueries: int = 3,
                k: int = 5):

        self.retriever = retriever
        self.llm_chain = llm_chain
        self.num_subqueries = num_subqueries
        self.k = k
        
        
    @classmethod
    def from_retrievers(cls,
                        base_retrievers: list[BaseRetriever],
                        weights: list[float] | None = None,
                        k: int = 5):
        
        if not base_retrievers:
            raise ValueError("At least one retriver must be provided.")
        
        if not  isinstance(base_retrievers, list):
            raise ValueError("base_retrievers must be a list of retriever instances")
        
        ensemble = EnsembleRetriever(retrievers=base_retrievers, weights=weights)
        
        return cls(retriever=ensemble, k=k)
    
    
    @classmethod
    def from_llm(cls,
                 llm,
                 retriever: BaseRetriever,
                 num_subqueries: int = 3,
                 k: int = 5):
        
        prompt = ChatPromptTemplate(
            messages=[
                ("system", "You are a helpful assistant that generates sub-queries to a main query to enhance retrieval"),
                ("user","Given the main query: '{main_query}', generate {num_subqueries} sub-queries that can be used to retrieve relevant documents")
            ],
            input_variables=['main_query', 'num_subqueries']
        )
        
        structured_llm = llm.with_structured_output(SubQuerySchema)
        
        llm_chain = prompt | structured_llm
        
        return cls(retriever=retriever, llm_chain=llm_chain, num_subqueries=num_subqueries, k=k)
    
    
    def _retrieve_documents(self, query: str) -> list[Document]:
        # internal method to retrieve the documents using the retriever
        return self.retriever.invoke(query)
    
    
    def _generate_subqueries(self, query: str) -> list[str]:
        # internal method to generate the sub-queries using the LLM
        if not self.llm_chain:
            raise ValueError("LLM chain is not provided for generating sub-queries")
        
        logger.info(f"Generating {self.num_subqueries} sub-queries for main query: {query}")
        
        result = self.llm_chain.invoke({"main_query": query, "num_subqueries": self.num_subqueries})
        
        logger.info(f"Generated sub-queries: {result.sub_queries}")
        return result.sub_queries
        
        
    def _reciprocal_rank_fusion(self, retrieved_docs: list[list[Document]]) -> list[Document]:
        # internal method to perform RRF on the retrieved documents.
        
        doc_scores: dict[str, tuple[float, Document]] = {}  # {"doc_text": (rrf_score, doc_object)}
        
        for retrieved_set in retrieved_docs:
            for rank, doc in enumerate(retrieved_set, start=1): # let [doc_2(rank=1), doc_4(rank=2), doc_1(rank=3)]
                
                rrf_score = 1.0 / (rank + 60)
                key = doc.page_content
                
                if key in doc_scores:  # if doc is in dict {"key": (rrf_score, doc_object)}
                    prev_score, prev_doc = doc_scores[key] # fetching score and doc_object
                    doc_scores[key] = (prev_score + rrf_score, prev_doc) # updating the RRF score
                else:
                    doc_scores[key] = (rrf_score, doc)
                    
        doc_with_scores = doc_scores.values()
        sorted_docs = sorted(doc_with_scores, key=lambda x: x[0], reverse=True)
        
        logger.info(f"RRF scores with documents: {doc_with_scores}")
        
        return [doc for _, doc in sorted_docs]
    
    
    def invoke(self, query: str) -> list[Document]:
        
        # main method to invoke the RAG fusion process
        
        if self.llm_chain:
            sub_queries = self._generate_subqueries(query=query)
            
            all_retrieved_docs = [self._retrieve_documents(sub_query) for sub_query in sub_queries]
            logger.info(f"Retrieved documents per sub_query: {[[doc.page_content[:50] for doc in docs] for docs in all_retrieved_docs]}")
            
            fused_docs = self._reciprocal_rank_fusion(all_retrieved_docs)
            return fused_docs[:self.k]
        else:
            return self._retrieve_documents(query=query)[:self.k]