import {
  LanguageModelV3,
  LanguageModelV3CallOptions,
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3ResponseMetadata,
  LanguageModelV3StreamPart,
  LanguageModelV3StreamResult,
  LanguageModelV3Usage,
  SharedV3Headers,
  SharedV3Warning
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createJsonResponseHandler,
  generateId,
  postJsonToApi
} from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import { convertToOllamaCompletionPrompt } from "../adaptors/convert-to-ollama-completion-prompt";
import { mapOllamaFinishReason } from "../adaptors/map-ollama-finish-reason";
import { getResponseMetadata } from "../common/get-response-metadata";
import { createNdjsonStreamResponseHandler } from "../common/ndjson-stream-handler";
import { OllamaCompletionModelId, OllamaCompletionSettings } from "./ollama-completion-settings";
import { ollamaFailedResponseHandler } from "./ollama-error";

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

export class OllamaCompletionLanguageModel implements LanguageModelV3 {
  readonly specificationVersion = "v3" as const;

  readonly modelId: OllamaCompletionModelId;
  readonly settings: OllamaCompletionSettings;
  readonly provider: string;
  readonly defaultObjectGenerationMode = undefined;
  readonly supportsImageUrls = false;

  private readonly config: OllamaCompletionConfig;

  constructor(
    modelId: OllamaCompletionModelId,
    settings: OllamaCompletionSettings,
    config: OllamaCompletionConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
    this.provider = config.provider;
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    // No URLs are supported for completion models.
  };

  private getArgs({
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
  }: LanguageModelV3CallOptions) {
    const warnings: SharedV3Warning[] = [];

    if (topK != null) {
      warnings.push({
        type: "unsupported",
        feature: "topK",
      });
    }

    if (tools?.length) {
      warnings.push({ type: "unsupported", feature: "tools" });
    }

    if (toolChoice != null) {
      warnings.push({ type: "unsupported", feature: "toolChoice" });
    }

    if (responseFormat != null && responseFormat.type !== "text") {
      warnings.push({
        type: "unsupported",
        feature: "responseFormat (JSON)",
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
        user: this.settings.user,
        think: this.settings.think,

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
        suffix: this.settings.suffix,
        echo: this.settings.echo,
        stream: false, // always disabled for doGenerate
      },
      warnings,
    };
  }

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
    const { args: body, warnings } = this.getArgs(options);

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
      failedResponseHandler: ollamaFailedResponseHandler as any,
      successfulResponseHandler: createJsonResponseHandler(
        baseOllamaResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const { prompt: rawPrompt, ...rawSettings } = body;

    const typedResponse = response as z.infer<typeof baseOllamaResponseSchema>;

    return {
      content: [
        {
          type: "text",
          text: typedResponse.response,
        },
      ],
      usage: {
        inputTokens: {
          total: typedResponse.prompt_eval_count ?? 0,
          noCache: undefined,
          cacheRead: undefined,
          cacheWrite: undefined,
        },
        outputTokens: {
          total: typedResponse.eval_count ?? 0,
          text: undefined,
          reasoning: undefined,
        },
      },
      finishReason: mapOllamaFinishReason("stop"),
      request: { body: JSON.stringify(body) },
      response: {
        ...getResponseMetadata(typedResponse),
        headers: responseHeaders,
        body: rawResponse,
      },
      warnings,
    };
  }

  async doStream(
    options: LanguageModelV3CallOptions,
  ): Promise<LanguageModelV3StreamResult & {warnings: Array<SharedV3Warning>}> {
    const { args, warnings } = this.getArgs(options);

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
      failedResponseHandler: ollamaFailedResponseHandler as any,
      successfulResponseHandler: createNdjsonStreamResponseHandler(
        baseOllamaResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const { prompt: rawPrompt, ...rawSettings } = args;

    let finishReason = "other";
    let usage: LanguageModelV3Usage = {
      inputTokens: {
        total: undefined,
        noCache: undefined,
        cacheRead: undefined,
        cacheWrite: undefined,
      },
      outputTokens: {
        total: undefined,
        text: undefined,
        reasoning: undefined,
      },
    };
    let isFirstChunk = true;
    let textStarted = false;
    const textId = generateId();

    return {
      stream: response.pipeThrough(
        new TransformStream<
          z.infer<typeof baseOllamaResponseSchema>,
          LanguageModelV3StreamPart
        >({
          transform(chunk, controller) {
            
            const value = chunk;

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
              finishReason = "stop";
            }

            if (value.response != null) {
              if (!textStarted) {
                controller.enqueue({
                  type: "text-start",
                  id: textId,
                });
                textStarted = true;
              }
              controller.enqueue({
                type: "text-delta",
                id: textId,
                delta: value.response,
              });
            }
          },

          flush(controller) {
            if (textStarted) {
              controller.enqueue({
                type: "text-end",
                id: textId,
              });
            }
            controller.enqueue({
              type: "finish",
              finishReason: mapOllamaFinishReason(finishReason),
              usage,
            });
          },
        }),
      ),
      request: { body: JSON.stringify(body) },
      response: { headers: responseHeaders },
      warnings: warnings
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


