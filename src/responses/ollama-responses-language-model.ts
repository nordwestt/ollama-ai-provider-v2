import {
  LanguageModelV2,
  LanguageModelV2CallWarning,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonResponseHandler,
  postJsonToApi,
  safeParseJSON,
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
import { z } from "zod/v4";
import { ParseResult } from "@ai-sdk/provider-utils";

// Custom handler for newline-delimited JSON streams
function createJsonStreamResponseHandler<T>(schema: z.ZodSchema<T>) {
  return async ({ response }: { url: string; requestBodyValues: unknown; response: Response }) => {
    if (!response.body) {
      throw new Error('Response body is null');
    }
    
    const stream = response.body
      .pipeThrough(new TextDecoderStream())
      .pipeThrough(new TransformStream({
        transform(chunk: string, controller) {
          const lines = chunk.split('\n');
          for (const line of lines) {
            if (line.trim()) {
              const result = safeParseJSON({ text: line.trim(), schema });
              controller.enqueue(result);
            }
          }
        }
      }));

    const responseHeaders: Record<string, string> = {};
    response.headers.forEach((value, key) => {
      responseHeaders[key] = value;
    });

    return {
      value: stream,
      responseHeaders,
    };
  };
}

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
    'image/*': [
      /^https?:\/\/.*$/
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
      stream: (response as ReadableStream<ParseResult<z.infer<typeof baseOllamaResponseSchema>>>).pipeThrough(
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
