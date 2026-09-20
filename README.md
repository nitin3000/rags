This repository stores python code based agent that uses Oracle database for vector database. It has the following process

1) injest_code.py: Walks the code directory tree for java, python, go and react and injests that into the Oracle Vector Store DB. 
2) pr_review_service.py: This file uses fastapi toopne a listening server /review that accepts the web hook call from gitbhub and makes a call to LLM and the Oracle Vector stores to generate the review comments.
 Subsequently it makes a call to the github reviews URL to post a review comment on a particular code file. It takes the following steps
1) It receives the pr_number and the repo name from the web hook
2) It iterates all the files and the patched differences in particular file
3) It generates the embedding for the patch from the LLM
4) It runs a vector query for those embeddings in the Oracle Vector DB based on the cosine similarity
5) It generates a prompt for LLM based on the vector search results
6) runs a chat query to the LLM
7) Posts the review comments to the pr file
