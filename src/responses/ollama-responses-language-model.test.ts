import { OllamaResponsesLanguageModel } from './ollama-responses-language-model';
import {
  TEST_PROMPT,
  TEST_TOOLS,
  TEST_MODEL_ID,
  createTestConfig,
  createMockServer,
  prepareJsonResponse,
  prepareErrorResponse,
  prepareStreamResponse,
} from './test-helpers/ollama-test-helpers';

describe('OllamaResponsesLanguageModel', () => {
  const testConfig = createTestConfig();
  const model = new OllamaResponsesLanguageModel(TEST_MODEL_ID, testConfig);
  const server = createMockServer();

  describe('Model Properties', () => {
    it('should have correct specification version', () => {
      expect(model.specificationVersion).toBe('v2');
    });

    it('should have correct model ID', () => {
      expect(model.modelId).toBe(TEST_MODEL_ID);
    });

    it('should have correct provider', () => {
      expect(model.provider).toBe('ollama.responses');
    });

    it('should support image URLs', () => {
      expect(model.supportedUrls['image/*']).toEqual([/^https?:\/\/.*$/]);
    });
  });

  describe('doGenerate', () => {
    describe('Basic Generation', () => {
      it('should generate text response', async () => {
        prepareJsonResponse(server);

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
          totalTokens: 30,
          reasoningTokens: undefined,
          cachedInputTokens: undefined,
        });
      });

      it('should expose raw response data', async () => {
        prepareJsonResponse(server, {
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
    });

    describe('Tool Calls', () => {
      it('should handle tool calls', async () => {
        prepareJsonResponse(server, {
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
        prepareJsonResponse(server, {
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
    });

    describe('Settings and Options', () => {
      it('should return warnings for unsupported settings', async () => {
        prepareJsonResponse(server);

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
        prepareJsonResponse(server, {
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

      it('should handle provider options', async () => {
        prepareJsonResponse(server);

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

      it('should handle thinking levels for GPT-OSS models', async () => {
        const gptOssModel = new OllamaResponsesLanguageModel('gpt-oss:7b', testConfig);
        prepareJsonResponse(server);

        const result = await gptOssModel.doGenerate({
          prompt: TEST_PROMPT,
          providerOptions: {
            ollama: {
              think: 'high',
            },
          },
        });

        expect(result.warnings).toEqual([]);
      });

      it('should convert boolean think to level for GPT-OSS models', async () => {
        const gptOssModel = new OllamaResponsesLanguageModel('gpt-oss:7b', testConfig);
        prepareJsonResponse(server);

        const result = await gptOssModel.doGenerate({
          prompt: TEST_PROMPT,
          providerOptions: {
            ollama: {
              think: true,
            },
          },
        });

        expect(result.warnings).toEqual([]);
      });

      it('should handle thinking levels for regular models', async () => {
        prepareJsonResponse(server);

        const result = await model.doGenerate({
          prompt: TEST_PROMPT,
          providerOptions: {
            ollama: {
              think: 'medium',
            },
          },
        });

        expect(result.warnings).toEqual([]);
      });
    });

    describe('Error Handling', () => {
      it('should handle API errors', async () => {
        prepareErrorResponse(server);

        await expect(
          model.doGenerate({
            prompt: TEST_PROMPT,
          })
        ).rejects.toThrow();
      });
    });
  });

  describe('doStream', () => {
    describe('Basic Streaming', () => {
      it('should handle basic streaming', async () => {
        prepareStreamResponse(server);

        const result = await model.doStream({
          prompt: TEST_PROMPT,
        });

        expect(result.response?.headers).toBeDefined();
        expect(result.request?.body).toBeDefined();
        expect(result.stream).toBeDefined();
      });

      it('should handle stream with warnings', async () => {
        prepareStreamResponse(server);

        const result = await model.doStream({
          prompt: TEST_PROMPT,
          topK: 50, // unsupported setting
        });

        expect(result.stream).toBeDefined();
      });
    });
  });
});