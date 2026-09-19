import os
import oracledb
from langchain_oracledb.vectorstores import OracleVS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class OraclePRReviewEngine:
    def __init__(self, openai_api_key: str):
        """Initializes connection channels to pull code context from Oracle DB."""
        # Setup embeddings to match the vectorized format in the DB
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=openai_api_key
        )
        
        # Initialize the target LLM optimized for structural reasoning
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0.1, openai_api_key=openai_api_key)
        
        # Connect to your active running Oracle container
        self.connection = oracledb.connect(
            user="sys",
            password="YourSecurePassword",
            dsn="localhost:1521/FREE",
            mode=oracledb.AUTH_MODE_SYSDBA
        )
        
        # Attach to the existing database table holding your codebase context
        self.vector_store = OracleVS(
            client=self.connection,
            embedding_function=self.embeddings,
            table_name="REPOSITORY_CONTEXT",
            distance_strategy="COSINE"
        )

    def generate_review(self, incoming_diff: str, language: str) -> str:
        """
        Queries the database for existing code context and generates 
        an inline PR review comment block based on codebase conventions.
        """
        print(f"[+] Searching Oracle DB for existing {language} code conventions...")
        
        # 1. Pull the 3 most semantically similar code snippets from your codebase context
        retrieved_docs = self.vector_store.similarity_search(
            query=incoming_diff,
            k=3
        )
        
        # Build context reference string, isolating fragments matching the language
        context_snippets = []
        for doc in retrieved_docs:
            if doc.metadata.get("language") == language.lower():
                context_snippets.append(f"File: {doc.metadata.get('file_path')}\n```\n{doc.page_content}\n```")
        
        context_str = "\n\n".join(context_snippets) if context_snippets else "No matching language context found in database."

        # 2. Configure the reasoning system prompt template
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", (
                "You are an automated Staff Software Engineer evaluating an incoming Git Pull Request diff.\n"
                "Review the snippet based on security vulnerabilities, logic bugs, performance anomalies, and conformity to existing patterns.\n\n"
                "EXISTING CODE PATTERNS FROM REPOSITORY CONTEXT:\n{context}\n\n"
                "CRITICAL OUTPUT CRITERIA:\n"
                "- Keep comments concise and direct.\n"
                "- If the code looks correct, output a clean approval message.\n"
                "- Output your review in clean Markdown formatting."
            )),
            ("user", "Review this proposed {language} modification diff:\n\n```\n{diff}\n```")
        ])

        # 3. Chain and execute
        review_chain = prompt_template | self.llm
        print("[+] Processing review with LLM agent...")
        response = review_chain.invoke({
            "context": context_str,
            "language": language,
            "diff": incoming_diff
        })
        
        return response.content

    def close(self):
        if self.connection:
            self.connection.close()

# =====================================================================
# Main Execution / Test Block
# =====================================================================
if __name__ == "__main__":
    # ⚠️ Replace with your actual OpenAI Key
    API_KEY = os.environ["OPENAI_API_KEY"]
        
    # Mocking a faulty incoming PR modification to test our engine
    mock_react_diff = """
    export const SearchFilters = () => {
      // Logic flaw: Missing dependency array will trigger endless network refetches
      useEffect(() => {
        fetch('/api/properties').then(res => res.json());
      }); 
      
      return <div className="filter-panel">Filters</div>;
    };
    """
    
    reviewer = OraclePRReviewEngine(openai_api_key=API_KEY)
    feedback = reviewer.generate_review(mock_react_diff, language="react")
    
    print("\n📝 === AUTOMATED PR REVIEW FEEDBACK === 📝\n")
    print(feedback)
    reviewer.close()
