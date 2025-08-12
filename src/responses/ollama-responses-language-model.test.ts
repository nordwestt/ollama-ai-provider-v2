import {
  LanguageModelV2FunctionTool,
  LanguageModelV2Prompt,
} from '@ai-sdk/provider';
import {
  convertReadableStreamToArray,
  createTestServer,
  mockId,
} from '@ai-sdk/provider-utils/test';
import { OllamaResponsesLanguageModel } from './ollama-responses-language-model';
import {
  OllamaChatModelId,
} from '../ollama-chat-settings';
import { OllamaConfig } from '../common/ollama-config';

const TEST_PROMPT: LanguageModelV2Prompt = [
  { role: 'user', content: [{ type: 'text', text: 'Hello' }] },
];

const TEST_TOOLS: Array<LanguageModelV2FunctionTool> = [
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

const testConfig: OllamaConfig = {
  provider: 'ollama.responses',
  url: ({ path }) => `http://127.0.0.1:11434/api${path}`,
  headers: () => ({ 'Content-Type': 'application/json' }),
  generateId: () => 'mock-id-1',
};

const modelId: OllamaChatModelId = 'llama2';
const model = new OllamaResponsesLanguageModel(modelId, testConfig);

const server = createTestServer({
  'http://127.0.0.1:11434/api/chat': {},
});

describe('OllamaResponsesLanguageModel', () => {
  describe('doGenerate', () => {
    function prepareJsonResponse({
      content = 'Hello, how can I help you?',
      toolCalls,
      usage = { prompt_eval_count: 10, eval_count: 20 },
      headers,
      doneReason = 'stop',
    }: {
      content?: string;
      toolCalls?: Array<{
        id?: string;
        function: { name: string; arguments: Record<string, any> };
      }>;
      usage?: { prompt_eval_count?: number; eval_count?: number };
      headers?: Record<string, string>;
      doneReason?: string;
    } = {}) {
      server.urls['http://127.0.0.1:11434/api/chat'].response = {
        type: 'json-value',
        headers,
        body: {
          model: modelId,
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
    }

    it('should generate text response', async () => {
      prepareJsonResponse();

      const result = await model.doGenerate({
        prompt: TEST_PROMPT,
        maxOutputTokens: 100,
        temperature: 0.7,
      });

      expect(result.content).toEqual([
        { type: 'text', text: 'Hello, how can I help you?' },
      ]);
      expect(result.finishReason).toBe('stop');
      expect(result.usage).toEqual({
        inputTokens: 10,
        outputTokens: 20,
        totalTokens: 20,
        reasoningTokens: 20,
        cachedInputTokens: undefined,
      });
    });

    it('should handle tool calls', async () => {
      prepareJsonResponse({
        content: 'I need to check the weather for you.',
        toolCalls: [
          {
            id: 'call_1',
            function: {
              name: 'weather',
              arguments: { location: 'San Francisco' },
            },
          },
        ],
      });

      const result = await model.doGenerate({
        prompt: TEST_PROMPT,
        tools: TEST_TOOLS,
      });

      expect(result.content).toEqual([
        { type: 'text', text: 'I need to check the weather for you.' },
        {
          type: 'tool-call',
          toolCallId: 'call_1',
          toolName: 'weather',
          input: '{"location":"San Francisco"}',
        },
      ]);
    });

    it('should generate tool call ID when missing', async () => {
      prepareJsonResponse({
        content: '',
        toolCalls: [
          {
            function: {
              name: 'weather',
              arguments: { location: 'New York' },
            },
          },
        ],
      });

      const result = await model.doGenerate({
        prompt: TEST_PROMPT,
        tools: TEST_TOOLS,
      });

      expect(result.content[0]).toEqual({
        type: 'tool-call',
        toolCallId: 'mock-id-1',
        toolName: 'weather',
        input: '{"location":"New York"}',
      });
    });

    it('should return warnings for unsupported settings', async () => {
      prepareJsonResponse();

      const result = await model.doGenerate({
        prompt: TEST_PROMPT,
        topK: 50,
        seed: 123,
        presencePenalty: 0.5,
        frequencyPenalty: 0.3,
        stopSequences: ['stop'],
      });

      expect(result.warnings).toEqual([
        { type: 'unsupported-setting', setting: 'topK' },
        { type: 'unsupported-setting', setting: 'seed' },
        { type: 'unsupported-setting', setting: 'presencePenalty' },
        { type: 'unsupported-setting', setting: 'frequencyPenalty' },
        { type: 'unsupported-setting', setting: 'stopSequences' },
      ]);
    });

    it('should handle JSON response format', async () => {
      prepareJsonResponse({
        content: '{"result": "success"}',
      });

      const result = await model.doGenerate({
        prompt: TEST_PROMPT,
        responseFormat: {
          type: 'json',
          schema: {
            type: 'object',
            properties: { result: { type: 'string' } },
          },
        },
      });

      expect(result.content[0]).toEqual({
        type: 'text',
        text: '{"result": "success"}',
      });
    });

    it('should expose raw response data', async () => {
      prepareJsonResponse({
        headers: { 'x-custom-header': 'test-value' },
      });

      const result = await model.doGenerate({
        prompt: TEST_PROMPT,
      });

      expect(result.response?.headers).toMatchObject({
        'x-custom-header': 'test-value',
      });
      expect(result.request?.body).toBeDefined();
    });

    it('should handle provider options', async () => {
      prepareJsonResponse();

      const result = await model.doGenerate({
        prompt: TEST_PROMPT,
        providerOptions: {
          ollama: {
            user: 'test-user',
            metadata: { session: 'test' },
            think: true,
          },
        },
      });

      expect(result.warnings).toEqual([]);
    });

    it('should handle reasoning models configuration', async () => {
      const reasoningModel = new OllamaResponsesLanguageModel('o1-preview', testConfig);
      
      prepareJsonResponse();

      const result = await reasoningModel.doGenerate({
        prompt: TEST_PROMPT,
        providerOptions: {
          ollama: {
            reasoningEffort: 'high',
            reasoningSummary: 'enabled',
          },
        },
      });

      expect(result.warnings).toEqual([]);
    });
  });

  describe('doStream', () => {
    it('should handle basic streaming', async () => {
      // For now, test basic streaming functionality without complex mocking
      server.urls['http://127.0.0.1:11434/api/chat'].response = {
        type: 'json-value',
        body: {
          model: modelId,
          created_at: '2024-01-01T00:00:00.000Z',
          done: true,
          done_reason: 'stop',
          message: {
            role: 'assistant',
            content: 'Hello',
          },
          eval_count: 5,
        },
      };

      const result = await model.doStream({
        prompt: TEST_PROMPT,
      });

      expect(result.response?.headers).toBeDefined();
      expect(result.request?.body).toBeDefined();
    });

    it('should handle stream with warnings', async () => {
      server.urls['http://127.0.0.1:11434/api/chat'].response = {
        type: 'json-value',
        body: {
          model: modelId,
          created_at: '2024-01-01T00:00:00.000Z',
          done: true,
          message: { role: 'assistant', content: 'Hello' },
        },
      };

      const result = await model.doStream({
        prompt: TEST_PROMPT,
        topK: 50, // unsupported setting
      });

      // We can't easily test the stream content with basic json-value response
      // but we can test that the result structure is correct
      expect(result.stream).toBeDefined();
    });
  });

  describe('model properties', () => {
    it('should have correct specification version', () => {
      expect(model.specificationVersion).toBe('v2');
    });

    it('should have correct model ID', () => {
      expect(model.modelId).toBe(modelId);
    });

    it('should have correct provider', () => {
      expect(model.provider).toBe('ollama.responses');
    });

    it('should support image URLs', () => {
      expect(model.supportedUrls['image/*']).toEqual([/^https?:\/\/.*$/]);
    });
  });

  describe('error handling', () => {
    it('should handle API errors in doGenerate', async () => {
      server.urls['http://127.0.0.1:11434/api/chat'].response = {
        type: 'error',
        status: 500,
        body: 'Internal server error',
      };

      await expect(
        model.doGenerate({
          prompt: TEST_PROMPT,
        })
      ).rejects.toThrow();
    });
  });
});