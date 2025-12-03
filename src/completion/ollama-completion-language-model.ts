import {
  combineHeaders,
  createJsonResponseHandler,
  createBinaryResponseHandler,
  generateId,
  parseProviderOptions,
  ParseResult,
  postJsonToApi,
  parseJsonEventStream,
} from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import {
  InvalidPromptError,
  LanguageModelV3,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3StreamPart,
  LanguageModelV3Usage,
  SharedV3ProviderMetadata,
  SharedV3Warning,
} from '@ai-sdk/provider';
import { OllamaConfig } from '../common/ollama-config.js';
import { ollamaFailedResponseHandler } from './ollama-error.js';
import { convertToOllamaCompletionPrompt } from '../adaptors/convert-to-ollama-completion-prompt.js';
import {
  OllamaCompletionModelId,
  OllamaCompletionSettings,
} from './ollama-completion-settings.js';
import { mapOllamaFinishReason } from '../adaptors/map-ollama-finish-reason.js';
import { getResponseMetadata } from '../common/get-response-metadata.js';

// Completion-specific provider options schema
const ollamaCompletionProviderOptions = z.object({
  think: z.boolean().optional(),
  user: z.string().optional(),
  suffix: z.string().optional(),
  echo: z.boolean().optional(),
});

type OllamaCompletionConfig = {
  provider: string;
  url: (options: { path: string; modelId: string }) => string;
  headers: () => Record<string, string | undefined>;
  fetch?: typeof fetch;
};

export type OllamaCompletionProviderOptions = z.infer<
  typeof ollamaCompletionProviderOptions
>;

export class OllamaCompletionLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = 'v3';

  readonly modelId: OllamaCompletionModelId;
  readonly settings: OllamaCompletionSettings;

  private readonly config: OllamaCompletionConfig;

  constructor(
    modelId: OllamaCompletionModelId,
    settings: OllamaCompletionSettings,
    config: OllamaCompletionConfig
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  get provider(): string {
    return this.config.provider;
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    // No URLs are supported for completion models.
  };

  private async getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences: userStopSequences,
    responseFormat,
    tools,
    toolChoice,
    seed,
    providerOptions,
  }: Parameters<LanguageModelV3['doGenerate']>[0]) {
    const warnings: SharedV3Warning[] = [];

    // Parse provider options
    const ollamaOptions =
      (await parseProviderOptions({
        provider: 'ollama',
        providerOptions,
        schema: ollamaCompletionProviderOptions,
      })) ?? {};

    if (topK != null) {
      warnings.push({
        type: 'unsupported',
        feature: 'topK',
      });
    }

    if (tools?.length) {
      warnings.push({ type: 'unsupported', feature: 'tools' });
    }

    if (toolChoice != null) {
      warnings.push({ type: 'unsupported', feature: 'toolChoice' });
    }

    if (responseFormat != null && responseFormat.type !== 'text') {
      warnings.push({
        type: 'unsupported',
        feature: 'responseFormat',
        details: 'JSON response format is not supported.',
      });
    }

    const { prompt: completionPrompt, stopSequences } =
      convertToOllamaCompletionPrompt({ prompt });

    const stop = [...(stopSequences ?? []), ...(userStopSequences ?? [])];

    return {
      args: {
        // model id:
        model: this.modelId,

        // Ollama-supported settings:
        user: ollamaOptions.user,
        think: ollamaOptions.think,

        // standardized settings:
        max_tokens: maxOutputTokens,
        temperature,
        top_p: topP,
        frequency_penalty: frequencyPenalty,
        presence_penalty: presencePenalty,
        stop,

        // prompt:
        prompt: completionPrompt,

        // other settings:
        suffix: ollamaOptions.suffix,
        echo: ollamaOptions.echo,
        stream: false, // always disabled for doGenerate
      },
      warnings,
    };
  }

  async doGenerate(
    options: Parameters<LanguageModelV3['doGenerate']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doGenerate']>>> {
    const { args: body, warnings } = await this.getArgs(options);

    const {
      responseHeaders,
      value: response,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: this.config.url({
        path: '/generate',
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

    const { prompt: rawPrompt, ...rawSettings } = body;

    const providerMetadata: SharedV3ProviderMetadata = { ollama: {} };

    return {
      content: [
        {
          type: 'text',
          text: response.response,
        },
      ],
      usage: {
        inputTokens: response.prompt_eval_count ?? undefined,
        outputTokens: response.eval_count ?? undefined,
        totalTokens:
          (response.prompt_eval_count ?? 0) + (response.eval_count ?? 0),
      },
      finishReason: mapOllamaFinishReason('stop'),
      request: { body: JSON.stringify(body) },
      response: {
        ...getResponseMetadata(response),
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
      providerMetadata,
    };
  }

  async doStream(
    options: Parameters<LanguageModelV3['doStream']>[0]
  ): Promise<Awaited<ReturnType<LanguageModelV3['doStream']>>> {
    const { args, warnings } = await this.getArgs(options);

    const body = {
      ...args,
      stream: true,
    };

    const { responseHeaders, value: binaryStream } = await postJsonToApi({
      url: this.config.url({
        path: '/generate',
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
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

    const { prompt: rawPrompt, ...rawSettings } = args;

    let finishReason: LanguageModelV3FinishReason = 'unknown';
    let usage: LanguageModelV3Usage = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    };
    let isFirstChunk = true;

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof baseOllamaResponseSchema>>,
          LanguageModelV3StreamPart
        >({
          transform(chunk, controller) {
            // handle failed chunk parsing / validation:
            if (!chunk.success) {
              finishReason = 'error';
              controller.enqueue({
                type: 'error',
                error: (chunk as any).error,
              });
              return;
            }

            const value = chunk.value;

            // handle error chunks:
            if ('error' in value) {
              finishReason = 'error';
              controller.enqueue({ type: 'error', error: value.error });
              return;
            }

            if (isFirstChunk) {
              isFirstChunk = false;

              controller.enqueue({
                type: 'response-metadata',
                ...getResponseMetadata(value),
              });
            }

            if (value.done) {
              finishReason = mapOllamaFinishReason('stop');
            }

            if (value.response != null) {
              controller.enqueue({
                type: 'text-delta',
                id: '0',
                delta: value.response,
              });
            }
          },

          flush(controller) {
            controller.enqueue({
              type: 'finish',
              finishReason,
              usage,
            });
          },
        })
      ),
      request: { body: JSON.stringify(body) },
      response: { headers: responseHeaders },
    };
  }
}

const baseOllamaResponseSchema = z.object({
  model: z.string(),
  created_at: z.string(),
  response: z.string(),
  done: z.boolean(),
  context: z.array(z.number()),

  eval_count: z.number().optional(),
  eval_duration: z.number().optional(),

  load_duration: z.number().optional(),
  total_duration: z.number().optional(),

  prompt_eval_count: z.number().optional(),
  prompt_eval_duration: z.number().optional(),
});
