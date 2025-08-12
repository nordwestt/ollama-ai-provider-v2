import {
  EmbeddingModelV2,
  TooManyEmbeddingValuesForCallError,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonResponseHandler,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import { OllamaConfig } from "../common/ollama-config";
import {
  OllamaEmbeddingModelId,
  OllamaEmbeddingSettings,
} from "./ollama-embedding-settings";
import { ollamaFailedResponseHandler } from "../completion/ollama-error";

export class OllamaEmbeddingModel implements EmbeddingModelV2<string> {
  readonly specificationVersion = "v2";
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
    config: OllamaConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  async doEmbed({
    values,
    headers,
    abortSignal,
    providerOptions,
  }: Parameters<EmbeddingModelV2<string>["doEmbed"]>[0]): Promise<
    Awaited<ReturnType<EmbeddingModelV2<string>["doEmbed"]>>
  > {
    if (values.length > this.maxEmbeddingsPerCall) {
      throw new TooManyEmbeddingValuesForCallError({
        provider: this.provider,
        modelId: this.modelId,
        maxEmbeddingsPerCall: this.maxEmbeddingsPerCall,
        values,
      });
    }

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
      body: {
        model: this.modelId,
        input: values,
      },
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        ollamaTextEmbeddingResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    return {
      embeddings: response.embeddings.map((item) => item),
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
