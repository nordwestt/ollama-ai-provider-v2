import {
  combineHeaders,
  createJsonResponseHandler,
  generateId,
  parseProviderOptions,
  ParseResult,
  postJsonToApi,
  safeParseJSON,
} from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import {
  InvalidPromptError,
  LanguageModelV2,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
  SharedV2ProviderMetadata,
} from "@ai-sdk/provider";
import { OllamaConfig } from "../common/ollama-config";
import { ollamaFailedResponseHandler } from "./ollama-error";
import { convertToOllamaCompletionPrompt } from "../adaptors/convert-to-ollama-completion-prompt";
import { OllamaCompletionModelId, OllamaCompletionSettings } from "./ollama-completion-settings";
import { mapOllamaFinishReason } from "../adaptors/map-ollama-finish-reason";
import { getResponseMetadata } from "../common/get-response-metadata";

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

export type OllamaCompletionProviderOptions = z.infer<typeof ollamaCompletionProviderOptions>;

export class OllamaCompletionLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2";

  readonly modelId: OllamaCompletionModelId;
  readonly settings: OllamaCompletionSettings;

  private readonly config: OllamaCompletionConfig;

  constructor(
    modelId: OllamaCompletionModelId,
    settings: OllamaCompletionSettings,
    config: OllamaCompletionConfig,
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
  }: Parameters<LanguageModelV2["doGenerate"]>[0]) {
    const warnings: LanguageModelV2CallWarning[] = [];

    // Parse provider options
    const ollamaOptions =
      (await parseProviderOptions({
        provider: "ollama",
        providerOptions,
        schema: ollamaCompletionProviderOptions,
      })) ?? {};

    if (topK != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "topK",
      });
    }

    if (tools?.length) {
      warnings.push({ type: "unsupported-setting", setting: "tools" });
    }

    if (toolChoice != null) {
      warnings.push({ type: "unsupported-setting", setting: "toolChoice" });
    }

    if (responseFormat != null && responseFormat.type !== "text") {
      warnings.push({
        type: "unsupported-setting",
        setting: "responseFormat",
        details: "JSON response format is not supported.",
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
    options: Parameters<LanguageModelV2["doGenerate"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doGenerate"]>>> {
    const { args: body, warnings } = await this.getArgs(options);

    const {
      responseHeaders,
      value: response,
      rawValue: rawResponse,
    } = await postJsonToApi({
      url: this.config.url({
        path: "/generate",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: { ...body, stream: false },
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        baseOllamaResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const { prompt: rawPrompt, ...rawSettings } = body;

    const providerMetadata: SharedV2ProviderMetadata = { ollama: {} };

    return {
      content: [
        {
          type: "text",
          text: response.response,
        },
      ],
      usage: {
        inputTokens: response.prompt_eval_count ?? undefined,
        outputTokens: response.eval_count ?? undefined,
        totalTokens: (response.prompt_eval_count ?? 0) + (response.eval_count ?? 0),
      },
      finishReason: mapOllamaFinishReason("stop"),
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
    options: Parameters<LanguageModelV2["doStream"]>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV2["doStream"]>>> {
    const { args, warnings } = await this.getArgs(options);

    const body = {
      ...args,
      stream: true,
    };

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/generate",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonStreamResponseHandler(
        baseOllamaResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const { prompt: rawPrompt, ...rawSettings } = args;

    let finishReason: LanguageModelV2FinishReason = "unknown";
    let usage: LanguageModelV2Usage = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    };
    let isFirstChunk = true;

    return {
      stream: (response as ReadableStream<ParseResult<z.infer<typeof baseOllamaResponseSchema>>>).pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof baseOllamaResponseSchema>>,
          LanguageModelV2StreamPart
        >({
          transform(chunk, controller) {
            // handle failed chunk parsing / validation:
            if (!chunk.success) {
              finishReason = "error";
              controller.enqueue({ type: "error", error: (chunk as any).error });
              return;
            }

            const value = chunk.value;

            // handle error chunks:
            if ("error" in value) {
              finishReason = "error";
              controller.enqueue({ type: "error", error: value.error });
              return;
            }

            if (isFirstChunk) {
              isFirstChunk = false;

              controller.enqueue({
                type: "response-metadata",
                ...getResponseMetadata(value),
              });
            }

            if (value.done) {
              finishReason = mapOllamaFinishReason("stop");
            }

            if (value.response != null) {
              controller.enqueue({
                type: "text-delta",
                id: "0",
                delta: value.response,
              });
            }
          },

          flush(controller) {
            controller.enqueue({
              type: "finish",
              finishReason,
              usage,
            });
          },
        }),
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
