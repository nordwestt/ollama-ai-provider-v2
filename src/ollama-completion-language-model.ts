import {
  LanguageModelV2,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
  SharedV2ProviderMetadata,
} from "@ai-sdk/provider";
import {
  FetchFunction,
  ParseResult,
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  parseProviderOptions,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import { convertToOllamaCompletionPrompt } from "./convert-to-ollama-completion-prompt";
import { mapOllamaCompletionLogProbs } from "./map-ollama-completion-logprobs";
import { mapOllamaFinishReason } from "./map-ollama-finish-reason";
import {
  OllamaCompletionModelId,
  OllamaCompletionSettings,
} from "./ollama-completion-settings";
import {
  ollamaErrorDataSchema,
  ollamaFailedResponseHandler,
} from "./ollama-error";
import { getResponseMetadata } from "./get-response-metadata";
import { ollamaProviderOptions } from "./ollama-chat-settings";

type OllamaCompletionConfig = {
  provider: string;
  compatibility: "strict" | "compatible";
  headers: () => Record<string, string | undefined>;
  url: (options: { modelId: string; path: string }) => string;
  fetch?: FetchFunction;
};

export class OllamaCompletionLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2";
  readonly modelId: OllamaCompletionModelId;
  readonly settings?: OllamaCompletionSettings;

  private readonly config: OllamaCompletionConfig;

  private get providerOptionsName(): string {
    return this.config.provider.split(".")[0].trim();
  }

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
        schema: ollamaProviderOptions,
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

        // model specific settings:
        logit_bias: ollamaOptions.logitBias,
        logprobs:
          ollamaOptions?.logprobs === true
            ? 0
            : ollamaOptions?.logprobs === false
              ? undefined
              : ollamaOptions?.logprobs,
        user: ollamaOptions.user,

        // standardized settings:
        max_tokens: maxOutputTokens,
        temperature,
        top_p: topP,
        frequency_penalty: frequencyPenalty,
        presence_penalty: presencePenalty,
        seed,
        think: providerOptions?.ollama?.think ?? this.settings?.think,

        // prompt:
        prompt: completionPrompt,

        // stop sequences:
        stop: stop.length > 0 ? stop : undefined,
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
        totalTokens: response.eval_count ?? undefined,
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

      // only include stream_options when in strict compatibility mode:
      stream_options:
        this.config.compatibility === "strict"
          ? { include_usage: true }
          : undefined,
    };

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/generate",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createEventSourceResponseHandler(
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
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof baseOllamaResponseSchema>>,
          LanguageModelV2StreamPart
        >({
          transform(chunk, controller) {
            // handle failed chunk parsing / validation:
            if (!chunk.success) {
              finishReason = "error";
              controller.enqueue({ type: "error", error: chunk.error });
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
