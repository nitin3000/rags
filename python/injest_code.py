import os
import hashlib
import oracledb
from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_oracledb.vectorstores import OracleVS
from langchain_openai import OpenAIEmbeddings

# Import the modern package processing elements
import tree_sitter_language_pack as tspack

class OracleCodeIngestionPipeline:
    def __init__(self, openai_api_key: str):
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small", 
            openai_api_key=openai_api_key
        )
        
        self.connection = oracledb.connect(
            user="sys",
            password="YourSecurePassword",
            dsn="localhost:1521/FREE",
            mode=oracledb.AUTH_MODE_SYSDBA
        )
        
        self.vector_store = OracleVS(
            client=self.connection,
            embedding_function=self.embeddings,
            table_name="REPOSITORY_CONTEXT",
            distance_strategy="COSINE"
        )
        
        self.extension_map = {
            ".py": "python",
            ".java": "java",
            ".go": "go",
            ".jsx": "tsx",   # Normalized to standard tree-sitter grammar names
            ".tsx": "tsx"
        }

    def compute_file_hash(self, file_content: str) -> str:
        return hashlib.md5(file_content.encode('utf-8')).hexdigest()

    def ast_chunk_code(self, source_code: str, language: str) -> List[str]:
        """
        Uses the modern language pack's syntax-aware chunking engine to 
        isolate clean code blocks natively without custom parsing loops.
        """
        try:
            # Use the library's ProcessConfig to instruct it to track structure
            config = tspack.ProcessConfig(language).all()
            result = tspack.process(source_code, config)
            
            # Map structural components out into text elements
            functional_chunks = []
            for element in result.structure:
                # Extracts individual methods/classes cleanly 
                if element.kind in ['function', 'class', 'method']:
                    functional_chunks.append(element.text)
            
            if not functional_chunks:
                return [source_code]
            return functional_chunks
            
        except Exception as chunk_error:
            # Absolute baseline fallback if structure mapping encounters system variance
            return [fragment for fragment in source_code.split("\n\n") if fragment.strip()]

    def execute_repository_ingestion(self, directory_root: str):
        bulk_documents_buffer = []
        batch_size_threshold = 50
        
        print(f"[+] Initializing code scanning pipeline at workspace path: {directory_root}")
        
        for working_root, _, targeted_files in os.walk(directory_root):
            if any(forbidden in working_root for forbidden in [".git", "node_modules", "dist", "__pycache__"]):
                continue
                
            for current_file in targeted_files:
                file_extension = os.path.splitext(current_file)[1].lower()
                
                if file_extension in self.extension_map:
                    target_language = self.extension_map[file_extension]
                    absolute_file_path = os.path.join(working_root, current_file)
                    
                    try:
                        with open(absolute_file_path, "r", encoding="utf-8") as file_stream:
                            raw_code_data = file_stream.read()
                        
                        content_hash = self.compute_file_hash(raw_code_data)
                        relative_repo_path = os.path.relpath(absolute_file_path, directory_root)
                        
                        # Generate semantic chunks using the updated engine block
                        syntactic_fragments = self.ast_chunk_code(raw_code_data, target_language)
                        
                        for index, segment in enumerate(syntactic_fragments):
                            document_node = Document(
                                page_content=segment,
                                metadata={
                                    "file_path": relative_repo_path,
                                    "file_name": current_file,
                                    "language": target_language,
                                    "chunk_index": index,
                                    "md5_checksum": content_hash
                                }
                            )
                            bulk_documents_buffer.append(document_node)
                        
                        if len(bulk_documents_buffer) >= batch_size_threshold:
                            self.vector_store.add_documents(bulk_documents_buffer)
                            print(f"[+] Transferred payload batch of {len(bulk_documents_buffer)} vector items.")
                            bulk_documents_buffer.clear()
                            
                        print(f"[✔] Successfully processed tracking logs for: {relative_repo_path}")
                        
                    except Exception as transaction_failure:
                        print(f"[-] Operation failure on target file {current_file}: {transaction_failure}")
                        
        if bulk_documents_buffer:
            self.vector_store.add_documents(bulk_documents_buffer)
            print(f"[+] Transferred final balance of {len(bulk_documents_buffer)} vector items.")
            
        print("[🎉] System Repository Context synchronization successfully finalized.")

    def close_pipeline_connections(self):
        if self.connection:
            self.connection.close()
            print("[+] Connection channels to Oracle cluster severed cleanly.")

if __name__ == "__main__":
    # 1. Provide your live OpenAI API key credentials
    API_KEY_CREDENTIAL = os.environ["OPENAI_API_KEY"]
        # 2. Provide the local file path to your Python, Java, React, or Go repository
    TARGET_WORKSPACE = r"C:\Users\nitin\property-management-main"
    
    # 3. Instantiate the pipeline and run the code sync straight into Oracle
    pipeline = OracleCodeIngestionPipeline(openai_api_key=API_KEY_CREDENTIAL)
    pipeline.execute_repository_ingestion(TARGET_WORKSPACE)
    pipeline.close_pipeline_connections()
