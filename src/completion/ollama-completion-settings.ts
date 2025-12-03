// https://ollama.com/library
export type OllamaCompletionModelId = string & {};

export interface OllamaCompletionSettings {
  /**
   * Enable or disable the model's thinking process. When enabled, the output will separate
   * the model's thinking from the model's output. When disabled, the model will not think
   * and directly output the content.
   *
   * Only supported by certain models like DeepSeek R1 and Qwen 3.
   */
  think?: boolean;

  /**
   * Echo back the prompt in addition to the completion.
   */
  echo?: boolean;

  /**
   * The suffix that comes after a completion of inserted text.
   */
  suffix?: string;

  /**
   * A unique identifier representing your end-user, which can help Ollama to
   * monitor and detect abuse.
   */
  user?: string;
}
