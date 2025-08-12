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