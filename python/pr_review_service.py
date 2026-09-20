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
    print(repo)
    print(pr_number)
    
    # 🔥 FORCE THE CORRECT HARDCODED HTTP PATH FORMAT
    files_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/files"
    
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    print(f"[⚙️] Outbound API Request -> {files_url}")
    response = requests.get(files_url, headers=headers)
    
    if response.status_code != 200:
        print(f"[❌] Failed to fetch PR files from GitHub (Status {response.status_code}): {response.text}")
        return

    pr_files = response.json()
    comments = []
    
    # Modified Establish connection to local Oracle 23ai database
    connection = oracledb.connect(
        user=ORACLE_USER, password=ORACLE_PASSWORD, dsn=ORACLE_DSN,
        mode=oracledb.AUTH_MODE_SYSDBA 
    )
    
    # 🔥 FIXED: Changed 'pr_files[0]' to 'pr_files' to loop over elements correctly
    for file in pr_files: 
        filename = file['filename']
        patch_diff = file.get('patch') 
        
        if not patch_diff:
            continue 
            
        print(f"[🔍] Vectorizing and analyzing code diff for: {filename}")
        
        # Vectorize the patch code using OpenAI
        embedding_res = ai_client.embeddings.create(
            input=[patch_diff], model="text-embedding-3-large"
        )
     #   print("EMBEDDING TYPE:", type(embedding_res.data))
     #   print("EMBEDDING CONTENT SAMPLE:", embedding_res.data[:2] if isinstance(embedding_res.data, list) else embedding_res.data)

        diff_vector = embedding_res.data[0].embedding
        
        # 1. Keep the standard query format (Optionally remove TO_VECTOR since it's redundant now)
        sql = """
            SELECT file_path, code_content 
            FROM code_knowledge_base 
            ORDER BY VECTOR_DISTANCE(code_embedding, :1, COSINE) 
            FETCH FIRST 2 ROWS ONLY
        """
        
        with connection.cursor() as cursor:
            # 🔥 THE EXACT FIX: Tell the driver to treat the incoming list parameter as a vector
            cursor.setinputsizes(oracledb.DB_TYPE_VECTOR)
            
            # Pass your raw python embedding list directly (do not wrap it)
            cursor.execute(sql, [diff_vector])
            historical_matches = cursor.fetchall()
                        
        db_context = "\n".join([f"File: {row[0]}\nCode Snippet:\n{row[1]}" for row in historical_matches])
        
        
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
        review_feedback = llm_res.choices[0].message.content.strip()
        
        
        if "LGTM" not in review_feedback:
            comments.append({
                "path": filename,
                "position": 1, 
                "body": review_feedback
            })
            
      
    if comments:
        # 🔥 THE EXACT PATH FIX: Ensure the API maps to /repos/{owner}/{repo}/pulls/{number}/reviews
        review_url = f"https://api.github.com/repos/{repo}/pulls/{pr_number}/reviews"
        
        review_payload = {
            "body": "🤖 **Oracle RAG-Engine Codebase Analysis Complete.** Below are architectural enhancements recommended based on your historical code repository models:",
            "event": "COMMENT",  # Can be "COMMENT", "APPROVE", or "REQUEST_CHANGES"
            "comments": comments
        }
        
        # Ensure your custom request headers explicitly state the API version required by GitHub
        headers = {
            "Authorization": f"Bearer {github_token}",
            "Accept": "application/vnd.github+json",  # Standard JSON payload format specifier
            "X-GitHub-Api-Version": "2022-11-28"       # Enforces modern endpoint routing rules
        }
        
        print(f"[⚙️] Submitting review payload matrix to PR #{pr_number}...")
        review_res = requests.post(review_url, json=review_payload, headers=headers)
        
        # If still returning 422, inspect GitHub's explicit error text to catch specific validation drops
        if review_res.status_code != 200 and review_res.status_code != 201:
            print(f"[❌] GitHub rejected payload verification (Status {review_res.status_code}): {review_res.text}")
        else:
            print(f"[🎉] Published review payload to PR #{pr_number}. Response Status: {review_res.status_code}")

        
    connection.close()


# 🔥 FLEXIBLE, NON-BLOCKING RAW ROUTE HANDLER
@app.post("/review")
async def trigger_pr_review(request: Request, background_tasks: BackgroundTasks):
    """Receives and safely isolates native raw GitHub webhook data streams."""
    try:
        raw_body = await request.body()
        payload = json.loads(raw_body.decode("utf-8"))
    except Exception as e:
        raise HTTPException(status_code=400, detail="Malformed JSON payload submission.")

    # Handle GitHub test ping operations without throwing a 422 crash error
    if "zen" in payload:
        print("[🔔] GitHub Webhook Ping Connection Verification Event Echoed Successfully!")
        return {"message": "Webhook link test successful. System is listening."}

    action = payload.get("action")
    pull_request = payload.get("pull_request")
    repository = payload.get("repository")

    # Guard clause to ignore events we don't care about
    if action not in ["opened", "synchronize"] or not pull_request or not repository:
        return {"message": f"Action '{action}' ignored. No code review evaluation required."}

    repo_full_name = repository.get("full_name")
    pr_num = pull_request.get("number")
    
    # Comment
    github_token = os.getenv("GITHUB_TOKEN", "your_fallback_token_here")

    print(f"[+] Received incoming GitHub Webhook for Repo: {repo_full_name} | PR: {pr_num}")
    
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

    # 1. Retrieve the AuthToken from the system environment....
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
