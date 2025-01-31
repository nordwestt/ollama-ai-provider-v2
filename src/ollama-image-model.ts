import { ImageModelV1, ImageModelV1CallWarning } from '@ai-sdk/provider';
import {
  combineHeaders,
  createJsonResponseHandler,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { z } from 'zod';
import { OllamaConfig } from './ollama-config';
import { ollamaFailedResponseHandler } from './ollama-error';
import {
  OllamaImageModelId,
  OllamaImageSettings,
  modelMaxImagesPerCall,
} from './ollama-image-settings';

interface OllamaImageModelConfig extends OllamaConfig {
  _internal?: {
    currentDate?: () => Date;
  };
}

export class OllamaImageModel implements ImageModelV1 {
  readonly specificationVersion = 'v1';

  get maxImagesPerCall(): number {
    return (
      this.settings.maxImagesPerCall ?? modelMaxImagesPerCall[this.modelId] ?? 1
    );
  }

  get provider(): string {
    return this.config.provider;
  }

  constructor(
    readonly modelId: OllamaImageModelId,
    private readonly settings: OllamaImageSettings,
    private readonly config: OllamaImageModelConfig,
  ) {}

  async doGenerate({
    prompt,
    n,
    size,
    aspectRatio,
    seed,
    providerOptions,
    headers,
    abortSignal,
  }: Parameters<ImageModelV1['doGenerate']>[0]): Promise<
    Awaited<ReturnType<ImageModelV1['doGenerate']>>
  > {
    const warnings: Array<ImageModelV1CallWarning> = [];

    if (aspectRatio != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'aspectRatio',
        details:
          'This model does not support aspect ratio. Use `size` instead.',
      });
    }

    if (seed != null) {
      warnings.push({ type: 'unsupported-setting', setting: 'seed' });
    }

    const currentDate = this.config._internal?.currentDate?.() ?? new Date();
    const { value: response, responseHeaders } = await postJsonToApi({
      url: this.config.url({
        path: '/images/generations',
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), headers),
      body: {
        model: this.modelId,
        prompt,
        n,
        size,
        ...(providerOptions.ollama ?? {}),
        response_format: 'b64_json',
      },
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        ollamaImageResponseSchema,
      ),
      abortSignal,
      fetch: this.config.fetch,
    });

    return {
      images: response.data.map(item => item.b64_json),
      warnings,
      response: {
        timestamp: currentDate,
        modelId: this.modelId,
        headers: responseHeaders,
      },
    };
  }
}

// minimal version of the schema, focussed on what is needed for the implementation
// this approach limits breakages when the API changes and increases efficiency
const ollamaImageResponseSchema = z.object({
  data: z.array(z.object({ b64_json: z.string() })),
});
