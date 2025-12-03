import { LanguageModelV3, SharedV3Warning } from '@ai-sdk/provider';
import {
  combineHeaders,
  createJsonResponseHandler,
  postJsonToApi,
  parseJsonEventStream,
} from '@ai-sdk/provider-utils';
import { OllamaConfig } from '../common/ollama-config.js';
import { ollamaFailedResponseHandler } from '../completion/ollama-error.js';
import { OllamaChatModelId } from '../ollama-chat-settings.js';
import {
  OllamaRequestBuilder,
  OllamaResponsesProviderOptions,
} from './ollama-responses-request-builder.js';
import { OllamaStreamProcessor } from './ollama-responses-stream-processor.js';
import {
  baseOllamaResponseSchema,
  OllamaResponseProcessor,
} from './ollama-responses-processor.js';

export class OllamaResponsesLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3';
  readonly modelId: OllamaChatModelId;

  private readonly config: OllamaConfig;
  private readonly requestBuilder: OllamaRequestBuilder;
  private readonly responseProcessor: OllamaResponseProcessor;

  constructor(modelId: OllamaChatModelId, config: OllamaConfig) {
    this.modelId = modelId;
    this.config = config;
    this.requestBuilder = new OllamaRequestBuilder();
    this.responseProcessor = new OllamaResponseProcessor(config);
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    'image/*': [/^https?:\/\/.*$/],
  };

  get provider(): string {
    return this.config.provider;
  }

  async doGenerate(
    options: Parameters<LanguageModelV3['doGenerate']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doGenerate']>>> {
    const { args: body, warnings } = await this.prepareRequest(options);

    const {
      responseHeaders,
      value: response,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: this.config.url({
        path: '/chat',
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: { ...body, stream: false },
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        baseOllamaResponseSchema
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const processedResponse =
      this.responseProcessor.processGenerateResponse(response);

    return {
      ...processedResponse,
      request: { body: JSON.stringify(body) },
      response: {
        modelId: this.modelId,
        timestamp: new Date(),
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
    };
  }

  async doStream(
    options: Parameters<LanguageModelV3['doStream']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doStream']>>> {
    const { args: body, warnings } = await this.prepareRequest(options);

    const { responseHeaders, value: binaryStream } = await postJsonToApi({
      url: this.config.url({
        path: '/chat',
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: { ...body, stream: true },
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: async ({ response }) => ({
        value: response.body!,
      }),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const response = parseJsonEventStream({
      stream: binaryStream,
      schema: baseOllamaResponseSchema,
    });

    const streamProcessor = new OllamaStreamProcessor(this.config);

    return {
      stream: response.pipeThrough(
        streamProcessor.createTransformStream(warnings, options)
      ),
      request: { body },
      response: { headers: responseHeaders },
    };
  }

  private async prepareRequest(
    options: Parameters<LanguageModelV3['doGenerate']>[0]
  ) {
    return await this.requestBuilder.buildRequest({
      modelId: this.modelId,
      ...options,
    });
  }
}

// Re-export types for convenience
export type { OllamaResponsesProviderOptions };
