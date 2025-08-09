import {
  InvalidResponseDataError,
  LanguageModelV2,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
  SharedV2ProviderMetadata,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
  createJsonStreamResponseHandler,
  generateId,
  parseProviderOptions,
  ParseResult,
  postJsonToApi,
} from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import { OllamaConfig } from "../ollama-config";
import { ollamaFailedResponseHandler } from "../ollama-error";
import { convertToOllamaResponsesMessages } from "./convert-to-ollama-responses-messages";
import { mapOllamaResponseFinishReason } from "./map-ollama-responses-finish-reason";
import { prepareResponsesTools } from "./ollama-responses-prepare-tools";
import { OllamaResponsesModelId } from "../ollama-responses-settings";
import { convertToOllamaChatMessages } from "../convert-to-ollama-chat-messages";
import { getResponseMetadata } from "../get-response-metadata";
import { mapOllamaFinishReason } from "../map-ollama-finish-reason";

export class OllamaResponsesLanguageModel implements LanguageModelV2 {
  readonly specificationVersion = "v2";

  readonly modelId: OllamaResponsesModelId;

  private readonly config: OllamaConfig;

  constructor(modelId: OllamaResponsesModelId, config: OllamaConfig) {
    this.modelId = modelId;
    this.config = config;
  }

  readonly supportedUrls: Record<string, RegExp[]> = {
    "image/*": [/^https?:\/\/.*$/],
  };

  get provider(): string {
    return this.config.provider;
  }

  private async getArgs({
    maxOutputTokens,
    temperature,
    stopSequences,
    topP,
    topK,
    presencePenalty,
    frequencyPenalty,
    seed,
    prompt,
    providerOptions,
    tools,
    toolChoice,
    responseFormat,
  }: Parameters<LanguageModelV2["doGenerate"]>[0]) {
    const warnings: LanguageModelV2CallWarning[] = [];
    const modelConfig = getResponsesModelConfig(this.modelId);

    if (topK != null) {
      warnings.push({ type: "unsupported-setting", setting: "topK" });
    }

    if (seed != null) {
      warnings.push({ type: "unsupported-setting", setting: "seed" });
    }

    if (presencePenalty != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "presencePenalty",
      });
    }

    if (frequencyPenalty != null) {
      warnings.push({
        type: "unsupported-setting",
        setting: "frequencyPenalty",
      });
    }

    if (stopSequences != null) {
      warnings.push({ type: "unsupported-setting", setting: "stopSequences" });
    }

    const { messages, warnings: messageWarnings } =
      convertToOllamaResponsesMessages({
        prompt,
        systemMessageMode: modelConfig.systemMessageMode,
      });

    warnings.push(...messageWarnings);
    const ollamaOptions = await parseProviderOptions({
      provider: "ollama",
      providerOptions,
      schema: ollamaResponsesProviderOptionsSchema,
    });
    const strictJsonSchema = ollamaOptions?.strictJsonSchema ?? false;
    const baseArgs = {
      model: this.modelId,
      messages: convertToOllamaChatMessages({
        prompt,
        systemMessageMode: "system",
      }),
      temperature,
      top_p: topP,
      max_output_tokens: maxOutputTokens,

      ...(responseFormat?.type === "json" && {
        text: {
          format:
            responseFormat.schema != null
              ? {
                  type: "json_schema",
                  strict: strictJsonSchema,
                  name: responseFormat.name ?? "response",
                  description: responseFormat.description,
                  schema: responseFormat.schema,
                }
              : { type: "json_object" },
        },
      }),

      // provider options:
      metadata: ollamaOptions?.metadata,
      parallel_tool_calls: ollamaOptions?.parallelToolCalls,
      previous_response_id: ollamaOptions?.previousResponseId,
      store: ollamaOptions?.store,
      user: ollamaOptions?.user,
      think: ollamaOptions?.think ?? false,
      instructions: ollamaOptions?.instructions,
      service_tier: ollamaOptions?.serviceTier,

      // model-specific settings:
      ...(modelConfig.isReasoningModel &&
        (ollamaOptions?.reasoningEffort != null ||
          ollamaOptions?.reasoningSummary != null) && {
          reasoning: {
            ...(ollamaOptions?.reasoningEffort != null && {
              effort: ollamaOptions.reasoningEffort,
            }),
            ...(ollamaOptions?.reasoningSummary != null && {
              summary: ollamaOptions.reasoningSummary,
            }),
          },
        }),
      ...(modelConfig.requiredAutoTruncation && {
        truncation: "auto",
      }),
    };
    if (modelConfig.isReasoningModel) {
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
    }
    // Validate flex processing support
    if (
      ollamaOptions?.serviceTier === "flex" &&
      !supportsFlexProcessing(this.modelId)
    ) {
      warnings.push({
        type: "unsupported-setting",
        setting: "serviceTier",
        details: "flex processing is only available for o3 and o4-mini models",
      });
      // Remove from args if not supported
      delete (baseArgs as any).service_tier;
    }
    const {
      tools: ollamaTools,
      toolChoice: ollamaToolChoice,
      toolWarnings,
    } = prepareResponsesTools({
      tools,
      toolChoice,
      strictJsonSchema,
    });

    console.log("Ollama tools:", ollamaTools);
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
    const { args: body, warnings } = await this.getArgs(options);

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: "/chat",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: {
        ...body,
        stream: true,
      },
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonStreamResponseHandler(
        baseOllamaResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const self = this;

    let finishReason: LanguageModelV2FinishReason = "unknown";
    let usage: LanguageModelV2Usage = {
      inputTokens: undefined,
      outputTokens: undefined,
      totalTokens: undefined,
    };
    let responseId: string | null = null;
    const ongoingToolCalls: Record<
      number,
      { toolName: string; toolCallId: string } | undefined
    > = {};
    let hasToolCalls = false;
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
              providerMetadata: {
                ollama: {
                  responseId,
                },
              },
            });
          },
        }),
      ),
      request: { body },
      response: { headers: responseHeaders },
    };
  }
}

const usageSchema = z.object({
  input_tokens: z.number(),
  input_tokens_details: z
    .object({ cached_tokens: z.number().nullish() })
    .nullish(),
  output_tokens: z.number(),
  output_tokens_details: z
    .object({ reasoning_tokens: z.number().nullish() })
    .nullish(),
});

const textDeltaChunkSchema = z.object({
  type: z.literal("response.output_text.delta"),
  item_id: z.string(),
  delta: z.string(),
});

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

const responseFinishedChunkSchema = z.object({
  type: z.enum(["response.completed", "response.incomplete"]),
  response: z.object({
    incomplete_details: z.object({ reason: z.string() }).nullish(),
    usage: usageSchema,
  }),
});

const responseCreatedChunkSchema = z.object({
  type: z.literal("response.created"),
  response: z.object({
    id: z.string(),
    created_at: z.number(),
    model: z.string(),
  }),
});

const responseOutputItemAddedSchema = z.object({
  type: z.literal("response.output_item.added"),
  output_index: z.number(),
  item: z.discriminatedUnion("type", [
    z.object({
      type: z.literal("message"),
      id: z.string(),
    }),
    z.object({
      type: z.literal("reasoning"),
      id: z.string(),
    }),
    z.object({
      type: z.literal("function_call"),
      id: z.string(),
      call_id: z.string(),
      name: z.string(),
      arguments: z.string(),
    }),
    z.object({
      type: z.literal("web_search_call"),
      id: z.string(),
      status: z.string(),
    }),
    z.object({
      type: z.literal("computer_call"),
      id: z.string(),
      status: z.string(),
    }),
  ]),
});

const responseOutputItemDoneSchema = z.object({
  type: z.literal("response.output_item.done"),
  output_index: z.number(),
  item: z.discriminatedUnion("type", [
    z.object({
      type: z.literal("message"),
      id: z.string(),
    }),
    z.object({
      type: z.literal("reasoning"),
      id: z.string(),
    }),
    z.object({
      type: z.literal("function_call"),
      id: z.string(),
      call_id: z.string(),
      name: z.string(),
      arguments: z.string(),
      status: z.literal("completed"),
    }),
    z.object({
      type: z.literal("web_search_call"),
      id: z.string(),
      status: z.literal("completed"),
    }),
    z.object({
      type: z.literal("computer_call"),
      id: z.string(),
      status: z.literal("completed"),
    }),
  ]),
});

const responseFunctionCallArgumentsDeltaSchema = z.object({
  type: z.literal("response.function_call_arguments.delta"),
  item_id: z.string(),
  output_index: z.number(),
  delta: z.string(),
});

const responseAnnotationAddedSchema = z.object({
  type: z.literal("response.output_text.annotation.added"),
  annotation: z.object({
    type: z.literal("url_citation"),
    url: z.string(),
    title: z.string(),
  }),
});

const responseReasoningSummaryTextDeltaSchema = z.object({
  type: z.literal("response.reasoning_summary_text.delta"),
  item_id: z.string(),
  output_index: z.number(),
  summary_index: z.number(),
  delta: z.string(),
});

const responseReasoningSummaryPartDoneSchema = z.object({
  type: z.literal("response.reasoning_summary_part.done"),
  item_id: z.string(),
  output_index: z.number(),
  summary_index: z.number(),
  part: z.unknown().nullish(),
});

const ollamaResponsesChunkSchema = z.union([
  textDeltaChunkSchema,
  responseFinishedChunkSchema,
  responseCreatedChunkSchema,
  responseOutputItemAddedSchema,
  responseOutputItemDoneSchema,
  responseFunctionCallArgumentsDeltaSchema,
  responseAnnotationAddedSchema,
  responseReasoningSummaryTextDeltaSchema,
  responseReasoningSummaryPartDoneSchema,
  z.object({ type: z.string() }).loose(), // fallback for unknown chunks
]);

function isTextDeltaChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof textDeltaChunkSchema> {
  return chunk?.type === "response.output_text.delta";
}

function isResponseOutputItemDoneChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseOutputItemDoneSchema> {
  return chunk?.type === "response.output_item.done";
}

function isResponseFinishedChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseFinishedChunkSchema> {
  return (
    chunk?.type === "response.completed" || chunk.type === "response.incomplete"
  );
}

function isResponseCreatedChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseCreatedChunkSchema> {
  return chunk?.type === "response.created";
}

function isResponseFunctionCallArgumentsDeltaChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseFunctionCallArgumentsDeltaSchema> {
  return chunk?.type === "response.function_call_arguments.delta";
}

function isResponseOutputItemAddedChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseOutputItemAddedSchema> {
  return chunk?.type === "response.output_item.added";
}

function isResponseAnnotationAddedChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseAnnotationAddedSchema> {
  return chunk?.type === "response.output_text.annotation.added";
}

function isResponseReasoningSummaryTextDeltaChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseReasoningSummaryTextDeltaSchema> {
  return chunk?.type === "response.reasoning_summary_text.delta";
}

function isResponseReasoningSummaryPartDoneChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseReasoningSummaryPartDoneSchema> {
  return chunk?.type === "response.reasoning_summary_part.done";
}

type ResponsesModelConfig = {
  isReasoningModel: boolean;
  systemMessageMode: "remove" | "system" | "developer";
  requiredAutoTruncation: boolean;
};

function getResponsesModelConfig(modelId: string): ResponsesModelConfig {
  // o series reasoning models:
  if (modelId.startsWith("o")) {
    if (modelId.startsWith("o1-mini") || modelId.startsWith("o1-preview")) {
      return {
        isReasoningModel: true,
        systemMessageMode: "remove",
        requiredAutoTruncation: false,
      };
    }

    return {
      isReasoningModel: true,
      systemMessageMode: "developer",
      requiredAutoTruncation: false,
    };
  }

  // gpt models:
  return {
    isReasoningModel: false,
    systemMessageMode: "system",
    requiredAutoTruncation: false,
  };
}

function supportsFlexProcessing(modelId: string): boolean {
  return modelId.startsWith("o3") || modelId.startsWith("o4-mini");
}

const ollamaResponsesProviderOptionsSchema = z.object({
  metadata: z.any().nullish(),
  parallelToolCalls: z.boolean().nullish(),
  previousResponseId: z.string().nullish(),
  store: z.boolean().nullish(),
  user: z.string().nullish(),
  reasoningEffort: z.string().nullish(),
  strictJsonSchema: z.boolean().nullish(),
  instructions: z.string().nullish(),
  reasoningSummary: z.string().nullish(),
  serviceTier: z.enum(["auto", "flex"]).nullish(),
  think: z.boolean().nullish(),
});

export type OllamaResponsesProviderOptions = z.infer<
  typeof ollamaResponsesProviderOptionsSchema
>;
