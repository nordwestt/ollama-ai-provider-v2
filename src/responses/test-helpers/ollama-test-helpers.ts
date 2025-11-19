import {
  LanguageModelV2FunctionTool,
  LanguageModelV2Prompt,
} from '@ai-sdk/provider';
import { http, HttpResponse } from 'msw';
import { setupServer } from 'msw/node';
import { OllamaChatModelId } from '../../ollama-chat-settings';
import { OllamaConfig } from '../../common/ollama-config';

export const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
];

export const TEST_TOOLS: Array<LanguageModelV2FunctionTool> = [
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

export const createMockServer = () => {
  const server = setupServer();
  
  return {
    server,
    listen: () => server.listen(),
    close: () => server.close(),
    resetHandlers: () => server.resetHandlers(),
    use: (...handlers: any[]) => server.use(...handlers),
  };
};

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
  const handler = http.post('http://127.0.0.1:11434/api/chat', () => {
    return HttpResponse.json({
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
    }, { headers });
  });
  
  server.use(handler);
};

export const prepareErrorResponse = (
  server: ReturnType<typeof createMockServer>,
  status: number = 500,
  body: string = 'Internal server error'
) => {
  const handler = http.post('http://127.0.0.1:11434/api/chat', () => {
    return new HttpResponse(body, { status });
  });
  
  server.use(handler);
};

export const prepareStreamResponse = (
  server: ReturnType<typeof createMockServer>,
  content: string = 'Hello'
) => {
  const handler = http.post('http://127.0.0.1:11434/api/chat', () => {
    return HttpResponse.json({
      model: TEST_MODEL_ID,
      created_at: '2024-01-01T00:00:00.000Z',
      done: true,
      done_reason: 'stop',
      message: {
        role: 'assistant',
        content,
      },
      eval_count: 5,
    });
  });
  
  server.use(handler);
};

// New MSW-specific helper functions for cleaner test code
export const createChatHandler = ({
  content = 'Hello, how can I help you?',
  toolCalls,
  usage = { prompt_eval_count: 10, eval_count: 20 },
  headers,
  doneReason = 'stop',
}: MockResponseOptions = {}) => {
  return http.post('http://127.0.0.1:11434/api/chat', () => {
    return HttpResponse.json({
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
    }, { headers });
  });
};

export const createErrorHandler = (
  status: number = 500,
  body: string = 'Internal server error'
) => {
  return http.post('http://127.0.0.1:11434/api/chat', () => {
    return new HttpResponse(body, { status });
  });
};

export const createEmbeddingHandler = (
  embeddings: number[][] = [[0.1, 0.2, 0.3, 0.4, 0.5], [0.6, 0.7, 0.8, 0.9, 1.0]],
  usage = { prompt_eval_count: 8 },
  headers?: Record<string, string>
) => {
  return http.post('http://127.0.0.1:11434/api/embed', () => {
    return HttpResponse.json({
      model: 'dummy-embedding-model',
      embeddings,
      total_duration: 14143917,
      load_duration: 1019500,
      prompt_eval_count: usage.prompt_eval_count,
    }, { headers });
  });
}; 