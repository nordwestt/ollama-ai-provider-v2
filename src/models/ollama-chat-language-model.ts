import {
  InvalidResponseDataError,
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
  SharedV2ProviderMetadata,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import {
  FetchFunction,
  ParseResult,
  combineHeaders,
  createJsonStreamResponseHandler,
  createJsonResponseHandler,
  generateId,
  isParsableJson,
  postJsonToApi,
  parseProviderOptions,
} from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import { convertToOllamaChatMessages } from "../adaptors/convert-to-ollama-chat-messages";
import { mapOllamaFinishReason } from "../adaptors/map-ollama-finish-reason";
import {
  OllamaChatModelId,
  ollamaProviderOptions,
} from "../ollama-chat-settings";
import {
  ollamaErrorDataSchema,
  ollamaFailedResponseHandler,
} from "../ollama-error";
import { getResponseMetadata } from "../get-response-metadata";
import { prepareTools } from "../ollama-prepare-tools";

type OllamaChatConfig = {
  provider: string;
  headers: () => Record<string, string | undefined>;
  url: (options: { modelId: string; path: string }) => string;
  fetch?: FetchFunction;
};

export class OllamaChatLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2";

  readonly modelId: OllamaChatModelId;
  readonly supportedUrls = {
    "image/*": [/^https?:\/\/.*$/],
  };

  private readonly config: OllamaChatConfig;

  constructor(modelId: OllamaChatModelId, config: OllamaChatConfig) {
    this.modelId = modelId;
    this.config = config;
  }

  get provider(): string {
    return this.config.provider;
  }

  private async getArgs({
    prompt,
    maxOutputTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences,
    responseFormat,
    seed,
    tools,
    toolChoice,
    providerOptions,
  }: LanguageModelV2CallOptions) {
    const warnings: LanguageModelV2CallWarning[] = [];

    // Parse provider options
    const ollamaOptions =
      (await parseProviderOptions({
        provider: "ollama",
        providerOptions,
        schema: ollamaProviderOptions,
      })) ?? {};

    const structuredOutputs = ollamaOptions.structuredOutputs ?? true;

    if (topK != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "topK",
      });
    }

    if (
      responseFormat?.type === "json" &&
      responseFormat.schema != null &&
      !structuredOutputs
    ) {
      warnings.push({
        type: "unsupported-setting",
        setting: "responseFormat",
        details:
          "JSON response format schema is only supported with structuredOutputs",
      });
    }

    const messages = convertToOllamaChatMessages({
      prompt,
      systemMessageMode: getSystemMessageMode(this.modelId),
    });

    //warnings.push(...messageWarnings);

    const strictJsonSchema = ollamaOptions.strictJsonSchema ?? false;

    const baseArgs = {
      // model id:
      model: this.modelId,

      // model specific settings:
      logit_bias: ollamaOptions.logitBias,
      logprobs:
        ollamaOptions.logprobs === true ||
        typeof ollamaOptions.logprobs === "number"
          ? true
          : undefined,
      top_logprobs:
        typeof ollamaOptions.logprobs === "number"
          ? ollamaOptions.logprobs
          : typeof ollamaOptions.logprobs === "boolean"
            ? ollamaOptions.logprobs
              ? 0
              : undefined
            : undefined,
      user: ollamaOptions.user,
      parallel_tool_calls: ollamaOptions.parallelToolCalls,

      // standardized settings:
      max_tokens: maxOutputTokens,
      temperature,
      top_p: topP,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      response_format:
        responseFormat?.type === "json"
          ? structuredOutputs && responseFormat.schema != null
            ? {
                type: "json_schema",
                json_schema: {
                  schema: responseFormat.schema,
                  strict: true,
                  name: responseFormat.name ?? "response",
                  description: responseFormat.description,
                },
              }
            : { type: "json_object" }
          : undefined,
      stop: stopSequences,
      seed,
      think: ollamaOptions.think ?? false,

      // messages:
      messages: convertToOllamaChatMessages({
        prompt,
        systemMessageMode: getSystemMessageMode(this.modelId),
      }),
    };

    if (isReasoningModel(this.modelId)) {
      // remove unsupported settings for reasoning models
      // see https://platform.ollama.com/docs/guides/reasoning#limitations
      if (baseArgs.temperature != null) {
        baseArgs.temperature = undefined;
        warnings.push({
          type: "unsupported-setting",
          setting: "temperature",
          details: "temperature is not supported for reasoning models",
        });
      }
      if (baseArgs.top_p != null) {
        baseArgs.top_p = undefined;
        warnings.push({
          type: "unsupported-setting",
          setting: "topP",
          details: "topP is not supported for reasoning models",
        });
      }
      if (baseArgs.frequency_penalty != null) {
        baseArgs.frequency_penalty = undefined;
        warnings.push({
          type: "unsupported-setting",
          setting: "frequencyPenalty",
          details: "frequencyPenalty is not supported for reasoning models",
        });
      }
      if (baseArgs.presence_penalty != null) {
        baseArgs.presence_penalty = undefined;
        warnings.push({
          type: "unsupported-setting",
          setting: "presencePenalty",
          details: "presencePenalty is not supported for reasoning models",
        });
      }
      if (baseArgs.logit_bias != null) {
        baseArgs.logit_bias = undefined;
        warnings.push({
          type: "other",
          message: "logitBias is not supported for reasoning models",
        });
      }
      if (baseArgs.logprobs != null) {
        baseArgs.logprobs = undefined;
        warnings.push({
          type: "other",
          message: "logprobs is not supported for reasoning models",
        });
      }
      if (baseArgs.top_logprobs != null) {
        baseArgs.top_logprobs = undefined;
        warnings.push({
          type: "other",
          message: "topLogprobs is not supported for reasoning models",
        });
      }
    }

    const {
      tools: ollamaTools,
      toolChoice: ollamaToolChoice,
      toolWarnings,
    } = prepareTools({
      tools,
      toolChoice,
      structuredOutputs,
      strictJsonSchema,
    });

    return {
      args: {
        ...baseArgs,
        tools: ollamaTools,
        tool_choice: ollamaToolChoice,
      },
      warnings: [...warnings, ...toolWarnings],
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
        path: "/chat",
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

    const { messages: rawPrompt, ...rawSettings } = body;

    const content: Array<LanguageModelV2Content> = [];

    const text = response.message.content;
    if (text != null && text.length > 0) {
      content.push({
        type: "text",
        text,
      });
    }

    // tool calls:
    for (const toolCall of response.message.tool_calls ?? []) {
      content.push({
        type: "tool-call" as const,
        toolCallId: toolCall.id ?? generateId(),
        toolName: toolCall.function.name,
        input: JSON.stringify(toolCall.function.arguments),
      });
    }

    // provider metadata:
    const providerMetadata: SharedV2ProviderMetadata = { ollama: {} };

    return {
      content,
      finishReason: mapOllamaFinishReason(response.done_reason),
      usage: {
        inputTokens: response.prompt_eval_count ?? undefined,
        outputTokens: response.eval_count ?? undefined,
        totalTokens: response.eval_count ?? undefined,
        reasoningTokens: response.eval_count ?? undefined,
        cachedInputTokens: undefined,
      },
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
        path: "/chat",
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

    const { messages: rawPrompt, ...rawSettings } = args;

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
          start(controller) {
            controller.enqueue({ type: "stream-start", warnings });
          },
          transform(chunk, controller) {
            // handle failed chunk parsing / validation:
            if (!chunk.success) {
              try {
                const text = (chunk.error as any).text as string;
                const lines = text.split("\n");
                lines.forEach((line) => {
                  if (line.trim() === "") return;
                  const parsed = JSON.parse(line);
                  controller.enqueue({
                    type: "text-delta",
                    id: "0",
                    delta: parsed.message.content,
                  });
                  controller.enqueue({
                    type: "reasoning-delta",
                    id: "0",
                    delta: parsed.message.thinking,
                  });
                });
                return;
              } catch (e) {
                console.error("ccchunk error", e);
              }
              console.error(
                "chunk error",
                Object.keys((chunk.error as any).text),
              );
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
              finishReason = mapOllamaFinishReason(value.done_reason);
              usage = {
                inputTokens: value.prompt_eval_count || 0,
                outputTokens: value.eval_count ?? undefined,
                totalTokens: value.eval_count ?? undefined,
              };
            }
            const delta = value?.message;

            if (delta?.content != null) {
              controller.enqueue({
                type: "text-delta",
                id: "0",
                delta: delta.content,
              });
            }

            if (delta?.thinking) {
              controller.enqueue({
                type: "reasoning-delta",
                id: "0",
                delta: delta.thinking,
              });
            }

            for (const toolCall of delta.tool_calls ?? []) {
              // Tool call start. Ollama returns all information except the arguments in the first chunk.
              if (toolCall.function?.name == null) {
                throw new InvalidResponseDataError({
                  data: toolCall,
                  message: `Expected 'function.name' to be a string.`,
                });
              }

              if (
                toolCall.function?.name != null &&
                toolCall.function?.arguments != null &&
                Object.keys(toolCall.function.arguments).length > 0
              ) {
                const id = generateId();

                controller.enqueue({
                  type: "tool-input-delta",
                  id: id,
                  delta: JSON.stringify(toolCall.function.arguments),
                });

                controller.enqueue({
                  type: "tool-call",
                  toolCallId: id,
                  toolName: toolCall.function.name,
                  input: JSON.stringify(toolCall.function.arguments),
                });
              }
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
  done: z.boolean(),
  message: z.object({
    content: z.string(),
    role: z.string(),
    thinking: z.string().optional(),
    tool_calls: z
      .array(
        z.object({
          function: z.object({
            name: z.string(),
            arguments: z.record(z.string(), z.any()),
          }),
          id: z.string().optional(),
        }),
      )
      .optional()
      .nullable(),
  }),

  done_reason: z.string().optional(),
  eval_count: z.number().optional(),
  eval_duration: z.number().optional(),
  load_duration: z.number().optional(),
  prompt_eval_count: z.number().optional(),
  prompt_eval_duration: z.number().optional(),
  total_duration: z.number().optional(),
});

function isReasoningModel(modelId: string) {
  return (
    modelId === "o1" ||
    modelId.startsWith("o1-") ||
    modelId === "o3" ||
    modelId.startsWith("o3-")
  );
}

function isAudioModel(modelId: string) {
  return modelId.startsWith("gpt-4o-audio-preview");
}

function getSystemMessageMode(modelId: string) {
  if (!isReasoningModel(modelId)) {
    return "system";
  }

  return (
    reasoningModels[modelId as keyof typeof reasoningModels]
      ?.systemMessageMode ?? "developer"
  );
}

function isStreamingSimulatedByDefault(modelId: string) {
  if (!isReasoningModel(modelId)) {
    return false;
  }

  return (
    reasoningModels[modelId as keyof typeof reasoningModels]
      ?.simulateStreamingByDefault ?? true
  );
}

const reasoningModels = {
  "o1-mini": {
    systemMessageMode: "remove",
    simulateStreamingByDefault: false,
  },
  "o1-mini-2024-09-12": {
    systemMessageMode: "remove",
    simulateStreamingByDefault: false,
  },
  "o1-preview": {
    systemMessageMode: "remove",
    simulateStreamingByDefault: false,
  },
  "o1-preview-2024-09-12": {
    systemMessageMode: "remove",
    simulateStreamingByDefault: false,
  },
} as const;
