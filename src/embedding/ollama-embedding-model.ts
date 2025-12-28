import {
  EmbeddingModelV3,
  TooManyEmbeddingValuesForCallError,
  SharedV3Warning,
  SharedV3Headers,
  SharedV3ProviderMetadata,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonResponseHandler,
  parseProviderOptions,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import { OllamaConfig } from "../common/ollama-config";
import {
  OllamaEmbeddingModelId,
  OllamaEmbeddingSettings,
} from "./ollama-embedding-settings";
import { ollamaFailedResponseHandler } from "../completion/ollama-error";

const ollamaEmbeddingProviderOptions = z.object({
  dimensions: z.number().optional(),
  truncate: z.boolean().optional(),
  keepAlive: z.string().optional(),
});

export type OllamaEmbeddingProviderOptions = z.infer<typeof ollamaEmbeddingProviderOptions>;

export class OllamaEmbeddingModel implements EmbeddingModelV3 {
  readonly specificationVersion = "v3" as const;
  readonly modelId: OllamaEmbeddingModelId;
  readonly provider: string;
  readonly maxEmbeddingsPerCall: number | undefined;
  readonly supportsParallelCalls: boolean;

  private readonly config: OllamaConfig;
  private readonly settings: OllamaEmbeddingSettings;

  constructor(
    modelId: OllamaEmbeddingModelId,
    settings: OllamaEmbeddingSettings,
    config: OllamaConfig
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
    this.provider = config.provider;
    this.maxEmbeddingsPerCall = settings.maxEmbeddingsPerCall ?? 2048;
    this.supportsParallelCalls = settings.supportsParallelCalls ?? true;
  }

  private getArgs({
    values,
  }: {
    values: Array<string>;
  }) {
    return {
      // model id:
      model: this.modelId,
      input: values,

      // advanced parameters:
      dimensions: this.settings.dimensions,
      truncate: this.settings.truncate,
      keep_alive: this.settings.keepAlive,
    };
  }

  async doEmbed({
    values,
    headers,
    abortSignal,
    providerOptions,
  }: {
    values: Array<string>;
    headers?: Record<string, string | undefined>;
    abortSignal?: AbortSignal;
    providerOptions?: Record<string, Record<string, unknown>>;
  }): Promise<{
    embeddings: Array<Array<number>>;
    usage?: { tokens: number };
    providerMetadata?: SharedV3ProviderMetadata;
    response?: {
      headers?: SharedV3Headers;
      body?: unknown;
    };
    warnings: Array<SharedV3Warning>;
  }> {
    if (this.maxEmbeddingsPerCall && values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values,
      });
    }

    const ollamaOptions = await parseProviderOptions({
      provider: 'ollama',
      providerOptions,
      schema: ollamaEmbeddingProviderOptions,
    });

    // Build body with explicit undefined filtering
    const dimensions = ollamaOptions?.dimensions ?? this.settings.dimensions;
    const truncate = ollamaOptions?.truncate ?? this.settings.truncate;
    const keepAlive = ollamaOptions?.keepAlive ?? this.settings.keepAlive;

    const body: Record<string, unknown> = {
      model: this.modelId,
      input: values,
    };

    if (dimensions !== undefined) body.dimensions = dimensions;
    if (truncate !== undefined) body.truncate = truncate;
    if (keepAlive !== undefined) body.keep_alive = keepAlive;

    const {
      responseHeaders,
      value: response,
      rawValue,
    } = await postJsonToApi({
      url: this.config.url({
        path: "/embed",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), headers),
      body: { ...body },
      failedResponseHandler: ollamaFailedResponseHandler as any,
      successfulResponseHandler: createJsonResponseHandler(
        ollamaTextEmbeddingResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    const typedResponse = response as z.infer<typeof ollamaTextEmbeddingResponseSchema>;

    return {
      embeddings: typedResponse.embeddings.map((item: number[]) => item),
      usage: { tokens: typedResponse.prompt_eval_count },
      response: { headers: responseHeaders, body: rawValue },
      warnings: [],
    };
  }
}

// minimal version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const ollamaTextEmbeddingResponseSchema = z.object({
  model: z.string(),
  embeddings: z.array(z.array(z.number())),
  total_duration: z.number(),
  load_duration: z.number(),
  prompt_eval_count: z.number(),
});
