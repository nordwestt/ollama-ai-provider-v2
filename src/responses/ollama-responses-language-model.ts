import {
  LanguageModelV2,
  LanguageModelV2CallWarning,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonResponseHandler,
  createJsonStreamResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { OllamaConfig } from "../common/ollama-config";
import { ollamaFailedResponseHandler } from "../completion/ollama-error";
import { OllamaChatModelId } from "../ollama-chat-settings";
import { 
  OllamaRequestBuilder,
  OllamaResponsesProviderOptions 
} from "./ollama-responses-request-builder";
import { 
  OllamaResponseProcessor, 
  baseOllamaResponseSchema 
} from "./ollama-responses-processor";
import { OllamaStreamProcessor } from "./ollama-responses-stream-processor";

export class OllamaResponsesLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2";
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
     image: [
      /^https?:\/\/.*\.(jpg|jpeg|png|gif|webp|bmp|svg)(\?.*)?$/i,
      /^data:image\/[^;]+;base64,/i,
    ]
  };

  get provider(): string {
    return this.config.provider;
  }

  async doGenerate(
    options: Parameters<LanguageModelV2["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doGenerate"]>>> {
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
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(baseOllamaResponseSchema),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const processedResponse = this.responseProcessor.processGenerateResponse(response);

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
    options: Parameters<LanguageModelV2["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doStream"]>>> {
    const { args: body, warnings } = await this.prepareRequest(options);

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/chat",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: { ...body, stream: true },
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonStreamResponseHandler(baseOllamaResponseSchema),
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
    };
  }

  private async prepareRequest(options: Parameters<LanguageModelV2["doGenerate"]>[0]) {
    return await this.requestBuilder.buildRequest({
      modelId: this.modelId,
      ...options,
    });
  }
}

// Re-export types for convenience
export type { OllamaResponsesProviderOptions };
