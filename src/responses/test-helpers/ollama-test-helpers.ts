import {
  LanguageModelV3FunctionTool,
  LanguageModelV3Prompt,
} from '@ai-sdk/provider';
import { createTestServer } from '../../test-utils/test-server';
import { OllamaChatModelId } from '../../ollama-chat-settings';
import { OllamaConfig } from '../../common/ollama-config';

export const TEST_PROMPT: LanguageModelV3Prompt = [
  { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
];

export const TEST_TOOLS: Array<LanguageModelV3FunctionTool> = [
  {
    type: 'function',
    name: 'weather',
    inputSchema: {
      type: 'object',
      properties: { location: { type: 'string' } },
      required: ['location'],
      additionalProperties: false,
    },
  },
  {
    type: 'function',
    name: 'cityAttractions',
    inputSchema: {
      type: 'object',
      properties: { city: { type: 'string' } },
      required: ['city'],
      additionalProperties: false,
    },
  },
];

export const TEST_MODEL_ID: OllamaChatModelId = 'llama2';

export const createTestConfig = (): OllamaConfig => ({
  provider: 'ollama.responses',
  url: ({ path }) => `http://127.0.0.1:11434/api${path}`,
  headers: () => ({ 'Content-Type': 'application/json' }),
  generateId: () => 'mock-id-1',
});

export const createMockServer : any = () =>
  createTestServer({
    'http://127.0.0.1:11434/api/chat': {},
  });

export interface MockResponseOptions {
  content?: string;
  toolCalls?: Array<{
    id?: string;
    function: { name: string; arguments: Record<string, any> };
  }>;
  usage?: { prompt_eval_count?: number; eval_count?: number };
  headers?: Record<string, string>;
  doneReason?: string;
}

export const prepareJsonResponse = (
  server: ReturnType<typeof createMockServer>,
  {
    content = 'Hello, how can I help you?',
    toolCalls,
    usage = { prompt_eval_count: 10, eval_count: 20 },
    headers,
    doneReason = 'stop',
  }: MockResponseOptions = {}
) => {
  server.urls['http://127.0.0.1:11434/api/chat'].response = {
    type: 'json-value',
    headers,
    body: {
      model: TEST_MODEL_ID,
      created_at: '2024-01-01T00:00:00.000Z',
      done: true,
      done_reason: doneReason,
      message: {
        role: 'assistant',
        content,
        tool_calls: toolCalls,
      },
      prompt_eval_count: usage.prompt_eval_count,
      eval_count: usage.eval_count,
    },
  };
};

export const prepareErrorResponse = (
  server: ReturnType<typeof createMockServer>,
  status: number = 500,
  body: string = 'Internal server error'
) => {
  server.urls['http://127.0.0.1:11434/api/chat'].response = {
    type: 'error',
    status,
    body,
  };
};

export const prepareStreamResponse = (
  server: ReturnType<typeof createMockServer>,
  content: string = 'Hello'
) => {
  server.urls['http://127.0.0.1:11434/api/chat'].response = {
    type: 'json-value',
    body: {
      model: TEST_MODEL_ID,
      created_at: '2024-01-01T00:00:00.000Z',
      done: true,
      done_reason: 'stop',
      message: {
        role: 'assistant',
        content,
      },
      eval_count: 5,
    },
  };
}; 