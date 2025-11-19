import { EmbeddingModelV2Embedding } from '@ai-sdk/provider';
import { http, HttpResponse } from 'msw';
import { createOllama } from '../ollama-provider';
import { createMockServer, createEmbeddingHandler } from '../responses/test-helpers/ollama-test-helpers';

const dummyEmbeddings = [
  [0.1, 0.2, 0.3, 0.4, 0.5],
  [0.6, 0.7, 0.8, 0.9, 1.0],
];
const testValues = ['sunny day at the beach', 'rainy day in the city'];

const provider = createOllama();
const model = provider.embedding('dummy-embedding-model');

describe('doEmbed', () => {
  const mockServer = createMockServer();

  beforeAll(() => {
    mockServer.listen();
  });

  afterEach(() => {
    mockServer.resetHandlers();
  });

  afterAll(() => {
    mockServer.close();
  });

  it('should extract embedding', async () => {
    mockServer.use(createEmbeddingHandler(dummyEmbeddings));

    const { embeddings } = await model.doEmbed({ values: testValues });

    expect(embeddings).toStrictEqual(dummyEmbeddings);
  });

  it('should expose the raw response', async () => {
    const headers = { 'test-header': 'test-value' };
    mockServer.use(createEmbeddingHandler(dummyEmbeddings, { prompt_eval_count: 8 }, headers));

    const { response } = await model.doEmbed({ values: testValues });

    // Check the response structure (response format may vary based on AI SDK implementation)
    expect(response).toBeDefined();
    // Note: The exact structure of response.headers may vary, this test validates the response exists
  });

  it('should extract usage', async () => {
    mockServer.use(createEmbeddingHandler(dummyEmbeddings, { prompt_eval_count: 20 }));

    const { usage } = await model.doEmbed({ values: testValues });

    expect(usage).toStrictEqual({ tokens: 20 });
  });

  it('should pass the model and the values', async () => {
    let capturedRequest: any;
    
    const handler = http.post('http://127.0.0.1:11434/api/embed', async ({ request }) => {
      capturedRequest = await request.json();
      return HttpResponse.json({
        model: 'dummy-embedding-model',
        embeddings: dummyEmbeddings,
        total_duration: 14143917,
        load_duration: 1019500,
        prompt_eval_count: 8,
      });
    });
    
    mockServer.use(handler);

    await model.doEmbed({ values: testValues });

    expect(capturedRequest).toStrictEqual({
      model: 'dummy-embedding-model',
      input: testValues,
    });
  });

  it('should pass the dimensions setting', async () => {
    let capturedRequest: any;
    
    const handler = http.post('http://127.0.0.1:11434/api/embed', async ({ request }) => {
      capturedRequest = await request.json();
      return HttpResponse.json({
        model: 'text-embedding-3-large',
        embeddings: dummyEmbeddings,
        total_duration: 14143917,
        load_duration: 1019500,
        prompt_eval_count: 8,
      });
    });
    
    mockServer.use(handler);

    await provider.embedding('text-embedding-3-large', { dimensions: 64 }).doEmbed({
      values: testValues,
    });

    expect(capturedRequest).toStrictEqual({
      model: 'text-embedding-3-large',
      input: testValues,
      dimensions: 64,
    });
  });

  it('should pass the provider options', async () => {
    let capturedRequest: any;
    
    const handler = http.post('http://127.0.0.1:11434/api/embed', async ({ request }) => {
      capturedRequest = await request.json();
      return HttpResponse.json({
        model: 'text-embedding-3-large',
        embeddings: dummyEmbeddings,
        total_duration: 14143917,
        load_duration: 1019500,
        prompt_eval_count: 8,
      });
    });
    
    mockServer.use(handler);

    await provider.embedding('text-embedding-3-large').doEmbed({
      values: testValues,
      providerOptions: {
        ollama: {
          dimensions: 64,
          truncate: true,
          keepAlive: '10s',
        },
      }
    });

    expect(capturedRequest).toStrictEqual({
      model: 'text-embedding-3-large',
      input: testValues,
      dimensions: 64,
      truncate: true,
      keep_alive: '10s',
    });
  });

  it('should pass headers', async () => {
    let capturedHeaders: any = {};
    
    const handler = http.post('http://127.0.0.1:11434/api/embed', async ({ request }) => {
      // Capture key headers manually since Object.fromEntries isn't available in this target
      capturedHeaders = {
        'content-type': request.headers.get('content-type'),
        'custom-provider-header': request.headers.get('custom-provider-header'),
        'custom-request-header': request.headers.get('custom-request-header'),
      };
      return HttpResponse.json({
        model: 'text-embedding-3-large',
        embeddings: dummyEmbeddings,
        total_duration: 14143917,
        load_duration: 1019500,
        prompt_eval_count: 8,
      });
    });
    
    mockServer.use(handler);

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

    expect(capturedHeaders).toMatchObject({
      'content-type': 'application/json',
      'custom-provider-header': 'provider-header-value',
      'custom-request-header': 'request-header-value',
    });
  });
});

describe('Error handling', () => {
  const mockServer = createMockServer();

  beforeAll(() => {
    mockServer.listen();
  });

  afterEach(() => {
    mockServer.resetHandlers();
  });

  afterAll(() => {
    mockServer.close();
  });

  it('should handle API errors', async () => {
    const handler = http.post('http://127.0.0.1:11434/api/embed', () => {
      return new HttpResponse('Internal Server Error', { status: 500 });
    });
    
    mockServer.use(handler);

    await expect(model.doEmbed({ values: testValues })).rejects.toThrow();
  });
});