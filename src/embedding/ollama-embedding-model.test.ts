import { EmbeddingModelV2Embedding } from '@ai-sdk/provider';
import { createTestServer } from '@ai-sdk/test-server/with-vitest';
import { createOllama } from '../ollama-provider.js';

const dummyEmbeddings = [
  [0.1, 0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9, 1.0],
];
const testValues = ['sunny day at the beach', 'rainy day in the city'];

const provider = createOllama();
const model = provider.embedding('dummy-embedding-model');

const server = createTestServer({
  'http://127.0.0.1:11434/api/embed': {},
});

describe('doEmbed', () => {
  function prepareJsonResponse({
    embeddings = dummyEmbeddings,
    usage = { prompt_eval_count: 8 },
    headers,
  }: {
    embeddings?: EmbeddingModelV2Embedding[];
    usage?: { prompt_eval_count: number };
    headers?: Record<string, string>;
  } = {}) {
    server.urls['http://127.0.0.1:11434/api/embed'].response = {
      type: 'json-value',
      headers,
      body: {
        model: 'dummy-embedding-model',
        embeddings,
        total_duration: 14143917,
        load_duration: 1019500,
        prompt_eval_count: usage.prompt_eval_count,
      },
    };
  }

  it('should extract embedding', async () => {
    prepareJsonResponse();

    const { embeddings } = await model.doEmbed({ values: testValues });

    expect(embeddings).toStrictEqual(dummyEmbeddings);
  });

  it('should expose the raw response', async () => {
    prepareJsonResponse({
      headers: {
        'test-header': 'test-value',
      },
    });

    const { response } = await model.doEmbed({ values: testValues });

    expect(response?.headers).toStrictEqual({
      // default headers:
      'content-length': '162',
      'content-type': 'application/json',

      // custom header
      'test-header': 'test-value',
    });
    expect(response).toMatchSnapshot();
  });

  it('should extract usage', async () => {
    prepareJsonResponse({
      usage: { prompt_eval_count: 20 },
    });

    const { usage } = await model.doEmbed({ values: testValues });

    expect(usage).toStrictEqual({ tokens: 20 });
  });

  it('should pass the model and the values', async () => {
    prepareJsonResponse();

    await model.doEmbed({ values: testValues });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: 'dummy-embedding-model',
      input: testValues,
    });
  });

  it('should pass the dimensions setting', async () => {
    prepareJsonResponse();

    await provider
      .embedding('text-embedding-3-large', { dimensions: 64 })
      .doEmbed({
        values: testValues,
      });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: 'text-embedding-3-large',
      input: testValues,
      dimensions: 64,
    });
  });

  it('should pass the provider options', async () => {
    prepareJsonResponse();

    await provider.embedding('text-embedding-3-large').doEmbed({
      values: testValues,
      providerOptions: {
        ollama: {
          dimensions: 64,
          truncate: true,
          keepAlive: '10s',
        },
      },
    });

    expect(await server.calls[0].requestBodyJson).toStrictEqual({
      model: 'text-embedding-3-large',
      input: testValues,
      dimensions: 64,
      truncate: true,
      keep_alive: '10s',
    });
  });

  it('should pass headers', async () => {
    prepareJsonResponse();

    const provider = createOllama({
      headers: {
        'Custom-Provider-Header': 'provider-header-value',
      },
    });

    await provider.embedding('text-embedding-3-large').doEmbed({
      values: testValues,
      headers: {
        'Custom-Request-Header': 'request-header-value',
      },
    });

    expect(server.calls[0].requestHeaders).toStrictEqual({
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });
});
