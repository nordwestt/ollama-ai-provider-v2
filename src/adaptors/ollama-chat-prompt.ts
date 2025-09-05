export type OllamaChatPrompt = Array<ChatCompletionMessage>;

export type ChatCompletionMessage =
  | ChatCompletionSystemMessage
  | ChatCompletionDeveloperMessage
  | ChatCompletionUserMessage
  | ChatCompletionAssistantMessage
  | ChatCompletionToolMessage
  | ChatCompletionFunctionMessage;

export interface ChatCompletionSystemMessage {
  role: 'system';
  content: string;
}

export interface ChatCompletionDeveloperMessage {
  role: 'developer';
  content: string;
}

export interface ChatCompletionUserMessage {
  role: 'user';
  content: string | Array<ChatCompletionContentPart>;
  images?: Array<Uint8Array | string | URL>;
}

export type ChatCompletionContentPart =
  | ChatCompletionContentPartText
  | ChatCompletionContentPartImage
  | ChatCompletionContentPartInputAudio;

export interface ChatCompletionContentPartText {
  type: 'text';
  text: string;
}

export interface ChatCompletionContentPartImage {
  type: 'image_url';
  image_url: { url: string };
}

export interface ChatCompletionContentPartInputAudio {
  type: 'input_audio';
  input_audio: { data: string; format: 'wav' | 'mp3' };
}

export interface ChatCompletionAssistantMessage {
  role: 'assistant';
  content?: string | null;
  tool_calls?: Array<ChatCompletionMessageToolCall>;
  thinking?: string;
  images?: Array<string>;
}

export interface ChatCompletionMessageToolCall {
  type: 'function';
  id: string;
  function: {
    arguments: object;
    name: string;
  };
}

export interface ChatCompletionToolMessage {
  role: 'tool';
  content: string;
  tool_call_id: string;
}

/**
 * Legacy function calling interface.
 * @deprecated this API is supported but deprecated by Ollama.
 */
export interface ChatCompletionFunctionMessage {
  role: 'function';
  content: string;
  name: string;
}
