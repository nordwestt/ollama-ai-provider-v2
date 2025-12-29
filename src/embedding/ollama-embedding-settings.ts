export type OllamaEmbeddingModelId =
  | 'text-embedding-3-small'
  | 'text-embedding-3-large'
  | 'text-embedding-ada-002'
  | (string & {});

export interface OllamaEmbeddingSettings {
  /**
Override the maximum number of embeddings per call.
   */
  maxEmbeddingsPerCall?: number;

  /**
Override the parallelism of embedding calls.
    */
  supportsParallelCalls?: boolean;

  /**
The number of dimensions the resulting output embeddings should have.
Only supported in text-embedding-3 and later models.
   */
  dimensions?: number;

  /**
A unique identifier representing your end-user, which can help Ollama to
monitor and detect abuse. Learn more.
*/
  user?: string;

  /**
Whether to truncate input text to fit within model's context length.
   */
  truncate?: boolean;

  /**
How long to keep the model loaded in memory (e.g., "5m" for 5 minutes).
   */
  keepAlive?: string;
}
