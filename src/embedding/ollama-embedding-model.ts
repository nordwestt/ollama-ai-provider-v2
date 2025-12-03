import {
  EmbeddingModelV3,
  TooManyEmbeddingValuesForCallError,
} from '@ai-sdk/provider';
import {
  combineHeaders,
  createJsonResponseHandler,
  parseProviderOptions,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import { OllamaConfig } from '../common/ollama-config.js';
import {
  OllamaEmbeddingModelId,
  OllamaEmbeddingSettings,
} from './ollama-embedding-settings.js';
import { ollamaFailedResponseHandler } from '../completion/ollama-error.js';

const ollamaEmbeddingProviderOptions = z.object({
  dimensions: z.number().optional(),
  truncate: z.boolean().optional(),
  keepAlive: z.string().optional(),
});

export type OllamaEmbeddingProviderOptions = z.infer<
  typeof ollamaEmbeddingProviderOptions
>;

export class OllamaEmbeddingModel implements EmbeddingModelV3 {
  readonly specificationVersion = 'v3';
  readonly modelId: OllamaEmbeddingModelId;

  private readonly config: OllamaConfig;
  private readonly settings: OllamaEmbeddingSettings;

  get provider(): string {
    return this.config.provider;
  }

  get maxEmbeddingsPerCall(): number {
    return this.settings.maxEmbeddingsPerCall ?? 2048;
  }

  get supportsParallelCalls(): boolean {
    return this.settings.supportsParallelCalls ?? true;
  }

  constructor(
    modelId: OllamaEmbeddingModelId,
    settings: OllamaEmbeddingSettings,
    config: OllamaConfig
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  private async getArgs({
    values,
    providerOptions,
  }: Parameters<EmbeddingModelV3['doEmbed']>[0]) {
    // Parse provider options
    const ollamaOptions =
      (await parseProviderOptions({
        provider: 'ollama',
        providerOptions,
        schema: ollamaEmbeddingProviderOptions,
      })) ?? {};

    return {
      args: {
        // model id:
        model: this.modelId,
        input: values,

        // advanced parameters:
        dimensions: ollamaOptions.dimensions ?? this.settings.dimensions,
        truncate: ollamaOptions.truncate,
        keep_alive: ollamaOptions.keepAlive,
      },
    };
  }

  async doEmbed({
    values,
    headers,
    abortSignal,
    providerOptions,
  }: Parameters<EmbeddingModelV3['doEmbed']>[0]): Promise<
    Awaited<ReturnType<EmbeddingModelV3['doEmbed']>>
  > {
    if (values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values,
      });
    }

    const { args: body } = await this.getArgs({ values, providerOptions });

    const {
      responseHeaders,
      value: response,
      rawValue,
    } = await postJsonToApi({
      url: this.config.url({
        path: '/embed',
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), headers),
      body: { ...body },
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        ollamaTextEmbeddingResponseSchema
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    return {
      embeddings: response.embeddings.map(item => item),
      usage: { tokens: response.prompt_eval_count },
      response: { headers: responseHeaders, body: rawValue },
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
