import os
import json
import requests
import oracledb
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="GitHub AI PR Reviewer - Oracle 23ai")

# 1. Initialize Clients (Set these in your environment variables)
# os.environ["OPENAI_API_KEY"] = "your-openai-key"
ai_client = OpenAI()

# Local Oracle 23ai Connection configuration
ORACLE_USER = os.getenv("ORACLE_USER", "system")
ORACLE_PASSWORD = os.getenv("ORACLE_PASSWORD", "your_password_here")
ORACLE_DSN = os.getenv("ORACLE_DSN", "localhost:1521/FREEPDB1") # Standard 23ai Free local DSN

from typing import Optional

# 1. Update the Pydantic models to mirror GitHub's native nesting
class RepositoryInfo(BaseModel):
    full_name: str # Maps to 'owner/repo_name'

class PullRequestInfo(BaseModel):
    number: int

class GitHubWebhookPayload(BaseModel):
    action: str # e.g., 'opened', 'synchronize'
    pull_request: Optional[PullRequestInfo] = None
    repository: RepositoryInfo


# 🟩 HEALTH CHECK ROUTE
@app.get("/health")
def health_check():
    """Endpoint to verify the local service is up and can connect to Oracle DB."""
    print(ORACLE_USER)
    print(ORACLE_PASSWORD)
    print(ORACLE_DSN)
    
    try:
        connection = oracledb.connect(
            user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN,
    mode=oracledb.AUTH_MODE_SYSDBA 
        )
        with connection.cursor() as cursor:
            cursor.execute("SELECT SYSDATE FROM DUAL")
            res = cursor.fetchone()
        connection.close()
        return {
            "status": "healthy",
            "database": "Oracle 23ai Connected",
            "timestamp": str(res[0])
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Database connection failed: {str(e)}")


def process_pr_review_workflow(repo: str, pr_number: int, github_token: str):
    """Orchestrates vector fetching, AI assessment, and GitHub publishing."""
    gh_base_url = "https://github.com"
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # Fetch pull request changed files & diff patches
    files_url = f"{gh_base_url}/repos/{repo}/pulls/{pr_number}/files"
    response = requests.get(files_url, headers=headers)
    if response.status_code != 200:
        print(f"[❌] Failed to fetch PR files from GitHub: {response.text}")
        return

    pr_files = response.json()
    comments = []
    
    # Establish connection to local Oracle 23ai database
    connection = oracledb.connect(user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN,
    mode=oracledb.AUTH_MODE_SYSDBA )
    
    for file in pr_files[0]:
        filename = file['filename']
        patch_diff = file.get('patch') # Extract the code lines changed
        
        if not patch_diff:
            continue # Skip binary files or massive additions without diff patches
            
        print(f"[🔍] Vectorizing and analyzing code diff for: {filename}")
        
        # Vectorize the patch code using OpenAI's standard text model
        embedding_res = ai_client.embeddings.create(
            input=[patch_diff], model="text-embedding-3-large"
        )
        diff_vector = embedding_res.data.embedding
        
        # Vector-search your local Oracle DB code repository context table
        with connection.cursor() as cursor:
            # Query utilizing Oracle 23ai's native VECTOR_DISTANCE feature
            sql = """
                SELECT file_path, code_content 
                FROM code_knowledge_base 
                ORDER BY VECTOR_DISTANCE(code_embedding, :1, COSINE) 
                FETCH FIRST 2 ROWS ONLY
            """
            cursor.execute(sql, [diff_vector])
            historical_matches = cursor.fetchall()
            
        # Format matching snippets to provide RAG anchor context
        db_context = "\n".join([f"File: {row[0]}\nCode Snippet:\n{row[1]}" for row in historical_matches])
        
        # Build prompt optimized for your PropTech platform rules
        prompt = f"""
        You are an elite Principal Technical Architect conducting a rigorous automated code review.
        Review this incoming pull request code diff:
        
        ### New Code Changes in {filename}:
        {patch_diff}
        
        ### Reference Architecture Patterns from your Local Knowledge Base:
        {db_context}
        
        Evaluate code quality, potential deadlocks/race conditions, anti-patterns, or missing component testing metrics.
        If the code looks perfect, respond exactly with 'LGTM'. Otherwise, output a concise 2-sentence maximum actionable feedback pointer.
        """
        
        llm_res = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        review_feedback = llm_res.choices.message.content.strip()
        
        if "LGTM" not in review_feedback:
            # Map review comments to the specific file patch line array location
            comments.append({
                "path": filename,
                "position": 1, # Line position relative to the diff chunk
                "body": review_feedback
            })
            
    # Post review comments bundle as a single cohesive review event block on GitHub
    if comments:
        review_url = f"{gh_base_url}/repos/{repo}/pulls/{pr_number}/reviews"
        review_payload = {
            "body": "🤖 **Oracle RAG-Engine Codebase Analysis Complete.** Below are architectural enhancements recommended based on your historical code repository models:",
            "event": "COMMENT",
            "comments": comments
        }
        review_res = requests.post(review_url, json=review_payload, headers=headers)
        print(f"[🎉] Published review payload to PR #{pr_number}. Response Status: {review_res.status_code}")
    else:
        print(f"[✔] PR #{pr_number} cleared cleanly with zero recommendations.")
        
    connection.close()


# 2. Update the route endpoint to parse the new payload
@app.post("/review")
def trigger_pr_review(payload: GitHubWebhookPayload, background_tasks: BackgroundTasks):
    """Receives native inbound GitHub webhook payloads seamlessly."""
    
    # Ignore PR actions that aren't opening or syncing code
    if payload.action not in ["opened", "synchronize"]:
        return {"message": f"Action '{payload.action}' ignored. No review required."}
        
    if not payload.pull_request:
        raise HTTPException(status_code=400, detail="Missing pull_request object context.")

    repo_full_name = payload.repository.full_name
    pr_num = payload.pull_request.number
    
    # ⚠️ CRITICAL CHANGE: Real webhooks don't send a personal token for security.
    # Retrieve your GitHub token securely from your environment variables instead.
    github_token = os.getenv("GITHUB_TOKEN", "your_fallback_token_here")

    print(f"[+] Received incoming GitHub Webhook for Repo: {repo_full_name} | PR: {pr_num}")
    
    # Queue the work to run asynchronously
    background_tasks.add_task(
        process_pr_review_workflow, 
        repo_full_name, 
        pr_num, 
        github_token
    )
    return {"message": f"GitHub Webhook processed successfully. PR #{pr_num} review queued."}
    
if __name__ == "__main__":
    import uvicorn
    import ngrok
    import sys

    # 1. Retrieve the AuthToken from the system environment
    auth_token = os.getenv("NGROK_AUTHTOKEN")
    
    if not auth_token:
        print("[❌] Error: NGROK_AUTHTOKEN environment variable is not set!")
        print("Please run: set NGROK_AUTHTOKEN=your_actual_token before executing.")
        sys.exit(1)

    print("[🌐] Connecting secure python-native tunnel to ngrok edge cloud...")
    
    # 2. Establish the cloud listener tunnel pointing to port 8000
    listener = ngrok.forward(
        addr=8000,
        authtoken=auth_token
    )
    
    print("\n" + "="*60)
    print(f"[🎉] PUBLIC GITHUB WEBHOOK URL ESTABLISHED!")
    print(f"👉 {listener.url()}/review")
    print("="*60 + "\n")
    
    # Keep the listener reference alive so it doesn't close on server threads
    app.state.ngrok_listener = listener

    # 3. Start the standard local FastAPI Uvicorn engine
    uvicorn.run("pr_review_service:app", host="127.0.0.1", port=8000, log_level="info")
