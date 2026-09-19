import os
import requests
from openai import OpenAI
import oracledb

# Initialize Clients
ai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gh_base_url = "https://github.com"

def handle_pr_review(repo, pr_number, github_token):
    headers = {
        "Authorization": f"Bearer {github_token}",
        "Accept": "application/vnd.github.v3+json"
    }
    
    # 1. Fetch changed files and diffs from the PR
    files_url = f"{gh_base_url}/repos/{repo}/pulls/{pr_number}/files"
    response = requests.get(files_url, headers=headers)
    pr_files = response.json()
    
    # Connect to your synced Oracle cluster
    connection = oracledb.connect(
        user="system", password="your_password", dsn="your_oracle_cloud_dsn"
    )
    
    comments = []
    
    for file in pr_files:
        filename = file['filename']
        patch_diff = file.get('patch') # This contains the actual lines changed
        
        if not patch_diff:
            continue
            
        # 2. Vectorize the incoming code diff line chunks
        embedding_res = ai_client.embeddings.create(
            input=[patch_diff], model="text-embedding-3-large"
        )
        diff_vector = embedding_res.data[0].embedding
        
        # 3. Vector-search Oracle DB for your PropTech repository context
        with connection.cursor() as cursor:
            # Query pulling historical implementations from your 141 vector items
            sql = """
                SELECT file_path, code_content 
                FROM code_knowledge_base 
                ORDER BY VECTOR_DISTANCE(code_embedding, :1, COSINE) 
                FETCH FIRST 2 ROWS ONLY
            """
            cursor.execute(sql, [diff_vector])
            historical_matches = cursor.fetchall()
            
        # Format matching context to anchor the LLM guidance
        db_context = "\n".join([f"From {row[0]}:\n{row[1]}" for row in historical_matches])
        
        # 4. Generate the Engineering/Architecture review feedback via LLM
        prompt = f"""
        You are a Principal Engineering Reviewer for a PropTech application codebase.
        Review this incoming pull request code diff:
        
        ### New Code Diff in {filename}:
        {patch_diff}
        
        ### Reference Patterns from existing Repository Context:
        {db_context}
        
        Identify any architectural inconsistencies, bugs, missing tests, or security concerns. 
        If it looks solid, return 'LGTM'. Otherwise, output a concise 2-sentence feedback point.
        """
        
        llm_res = ai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        review_feedback = llm_res.choices[0].message.content
        
        if "LGTM" not in review_feedback:
            # Add comment map to send back to GitHub review payload
            comments.append({
                "path": filename,
                "position": 1, # Line position relative to the diff patch
                "body": review_feedback
            })
            
    # 5. Post the synthesized review back onto the GitHub PR
    if comments:
        review_url = f"{gh_base_url}/repos/{repo}/pulls/{pr_number}/reviews"
        review_payload = {
            "body": "🤖 **Automated Architecture & Risk Review Complete.** Below are recommended enhancements based on historical codebase patterns in Oracle DB:",
            "event": "COMMENT",
            "comments": comments
        }
        requests.post(review_url, json=review_payload, headers=headers)
        
    connection.close()
    return {"status": "Review finalized successfully"}
