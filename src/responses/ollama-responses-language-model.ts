import {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3ResponseMetadata,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
  SharedV3Headers,
  SharedV3Warning,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { createNdjsonStreamResponseHandler } from "../common/ndjson-stream-handler";
import { OllamaConfig } from "../common/ollama-config";
import { ollamaFailedResponseHandler } from "../completion/ollama-error";
import { OllamaChatModelId } from "../ollama-chat-settings";
import {
  OllamaResponseProcessor,
  OllamaResponse,
  baseOllamaResponseSchema
} from "./ollama-responses-processor";
import {
  OllamaRequestBuilder,
  OllamaResponsesProviderOptions
} from "./ollama-responses-request-builder";
import { OllamaStreamProcessor } from "./ollama-responses-stream-processor";

export class OllamaResponsesLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly modelId: OllamaChatModelId;
  readonly provider: string;
  readonly defaultObjectGenerationMode = undefined;
  readonly supportsImageUrls = true;

  private readonly config: OllamaConfig;
  private readonly requestBuilder: OllamaRequestBuilder;
  private readonly responseProcessor: OllamaResponseProcessor;

  constructor(modelId: OllamaChatModelId, config: OllamaConfig) {
    this.modelId = modelId;
    this.config = config;
    this.provider = config.provider;
    this.requestBuilder = new OllamaRequestBuilder();
    this.responseProcessor = new OllamaResponseProcessor(config);
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    'image/*': [
      /^https?:\/\/.*$/
    ]
  };

  async doGenerate(
    options: LanguageModelV3CallOptions,
  ): Promise<{
    content: Array<LanguageModelV3Content>;
    finishReason: LanguageModelV3FinishReason;
    usage: LanguageModelV3Usage;
    warnings: Array<SharedV3Warning>;
    request?: { body?: unknown };
    response?: LanguageModelV3ResponseMetadata & {
      headers?: SharedV3Headers;
      body?: unknown;
    };
  }> {
    const { args: body, warnings } = await this.prepareRequest(options);

    const {
      responseHeaders,
      value: response,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: this.config.url({
        path: "/chat",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: { ...body, stream: false },
      failedResponseHandler: ollamaFailedResponseHandler as any,
      successfulResponseHandler: createJsonResponseHandler(baseOllamaResponseSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const processedResponse = this.responseProcessor.processGenerateResponse(response as OllamaResponse);

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
    options: LanguageModelV3CallOptions,
  ): Promise<{
    stream: ReadableStream<LanguageModelV3StreamPart>;
    warnings: Array<SharedV3Warning>;
    request?: { body?: unknown };
    response?: LanguageModelV3ResponseMetadata & {
      headers?: SharedV3Headers;
      body?: unknown;
    };
  }> {
    const { args: body, warnings } = await this.prepareRequest(options);

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/chat",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: { ...body, stream: true },
      failedResponseHandler: ollamaFailedResponseHandler as any,
      successfulResponseHandler: createNdjsonStreamResponseHandler(baseOllamaResponseSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const streamProcessor = new OllamaStreamProcessor(this.config);

    return {
      stream: response.pipeThrough(
        streamProcessor.createTransformStream(warnings, options)
      ),
      request: { body },
      response: { headers: responseHeaders },
      warnings: warnings
    };
  }

  private async prepareRequest(options: LanguageModelV3CallOptions) {
    return await this.requestBuilder.buildRequest({
      modelId: this.modelId,
      ...options,
    });
  }
}

// Re-export types for convenience
export type { OllamaResponsesProviderOptions };
