Prompt: RAG Pipeline

Create a simple local RAG pipeline in Python with clearly defined functions for each step so that the code is easy to read and follow and that we can later replac e the approach in each step with more sophisticated approaches.
Use the following naive RAG structure:
1.	Ingestion step: Read files in a dedicated folder inside of our project - this can then later be replaced with for instance a AWS S3 bucket
2.	Chunking step: For each document ingested, we need to chunk the document. Please recommend a simple approach to chunking to get us started.
3.	Embedding step: We need to create embeddings (for simplicity, let us use OpenAI embeddings for this step)
4.	Storage step: Let us for simplicity use an in memory vector db (for instance persisting it to a json file or similar. This will later be replaced by a proper vector database.)
5.	Retrieval step: Let us also create a retrieval function for testing using a basic cos similarity measure between a user query and the vector storage. We should also include a re-ranking function that can be empty for now, but that we can later implement.
6.	Evaluation step: Add a simple function for evaluation
We should be able to run steps 1-4 as one function (for ingesting and creating our knowledge base). And then subject to an established knowledge base, we shoul be able to run and test steps 5 and 6 separately.

