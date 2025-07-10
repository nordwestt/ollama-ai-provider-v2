import {
  LanguageModelV2,
  LanguageModelV2CallWarning,
  LanguageModelV2Content,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
} from "@ai-sdk/provider";
import {
  combineHeaders,
  createEventSourceResponseHandler,
  createJsonResponseHandler,
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
    console.log("getArgs1", this.modelId);
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
    console.log("getArgs2", this.modelId);

    const { messages, warnings: messageWarnings } =
      convertToOllamaResponsesMessages({
        prompt,
        systemMessageMode: modelConfig.systemMessageMode,
      });

    warnings.push(...messageWarnings);
    console.log("getArgs3", this.modelId);
    const ollamaOptions = await parseProviderOptions({
      provider: "ollama",
      providerOptions,
      schema: ollamaResponsesProviderOptionsSchema,
    });
    console.log("getArgs4", this.modelId);
    const strictJsonSchema = ollamaOptions?.strictJsonSchema ?? false;
    console.log("getArgs5", this.modelId);
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
    console.log("getArgs6", this.modelId);
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
    console.log("getArgs7", this.modelId);
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
    console.log("getArgs8", this.modelId);
    const {
      tools: ollamaTools,
      toolChoice: ollamaToolChoice,
      toolWarnings,
    } = prepareResponsesTools({
      tools,
      toolChoice,
      strictJsonSchema,
    });
    console.log("getArgs9", this.modelId);
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
        path: "/responses",
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body,
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        z.object({
          id: z.string(),
          created_at: z.number(),
          model: z.string(),
          output: z.array(
            z.discriminatedUnion("type", [
              z.object({
                type: z.literal("message"),
                role: z.literal("assistant"),
                content: z.array(
                  z.object({
                    type: z.literal("output_text"),
                    text: z.string(),
                    annotations: z.array(
                      z.object({
                        type: z.literal("url_citation"),
                        start_index: z.number(),
                        end_index: z.number(),
                        url: z.string(),
                        title: z.string(),
                      }),
                    ),
                  }),
                ),
              }),
              z.object({
                type: z.literal("function_call"),
                call_id: z.string(),
                name: z.string(),
                arguments: z.string(),
              }),
              z.object({
                type: z.literal("web_search_call"),
                id: z.string(),
                status: z.string().optional(),
              }),
              z.object({
                type: z.literal("computer_call"),
                id: z.string(),
                status: z.string().optional(),
              }),
              z.object({
                type: z.literal("reasoning"),
                summary: z.array(
                  z.object({
                    type: z.literal("summary_text"),
                    text: z.string(),
                  }),
                ),
              }),
            ]),
          ),
          incomplete_details: z.object({ reason: z.string() }).nullable(),
          usage: usageSchema,
        }),
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const content: Array<LanguageModelV2Content> = [];

    // map response content to content array
    for (const part of response.output) {
      switch (part.type) {
        case "reasoning": {
          content.push({
            type: "reasoning",
            text: part.summary
              .map((summary: { text: string }) => summary.text)
              .join(),
          });
          break;
        }

        case "message": {
          for (const contentPart of part.content) {
            content.push({
              type: "text",
              text: contentPart.text,
            });

            for (const annotation of contentPart.annotations) {
              content.push({
                type: "source",
                sourceType: "url",
                id: this.config.generateId?.() ?? generateId(),
                url: annotation.url,
                title: annotation.title,
              });
            }
          }
          break;
        }

        case "function_call": {
          content.push({
            type: "tool-call",
            toolCallId: part.call_id,
            toolName: part.name,
            input: part.arguments,
          });
          break;
        }

        case "web_search_call": {
          content.push({
            type: "tool-call",
            toolCallId: part.id,
            toolName: "web_search_preview",
            input: "",
            providerExecuted: true,
          });

          content.push({
            type: "tool-result",
            toolCallId: part.id,
            toolName: "web_search_preview",
            result: { status: part.status || "completed" },
            providerExecuted: true,
          });
          break;
        }

        case "computer_call": {
          content.push({
            type: "tool-call",
            toolCallId: part.id,
            toolName: "computer_use",
            input: "",
            providerExecuted: true,
          });

          content.push({
            type: "tool-result",
            toolCallId: part.id,
            toolName: "computer_use",
            result: {
              type: "computer_use_tool_result",
              status: part.status || "completed",
            },
            providerExecuted: true,
          });
          break;
        }
      }
    }

    return {
      content,
      finishReason: mapOllamaResponseFinishReason({
        finishReason: response.incomplete_details?.reason,
        hasToolCalls: content.some((part) => part.type === "tool-call"),
      }),
      usage: {
        inputTokens: response.usage.input_tokens,
        outputTokens: response.usage.output_tokens,
        totalTokens: response.usage.input_tokens + response.usage.output_tokens,
        reasoningTokens:
          response.usage.output_tokens_details?.reasoning_tokens ?? undefined,
        cachedInputTokens:
          response.usage.input_tokens_details?.cached_tokens ?? undefined,
      },
      request: { body },
      response: {
        id: response.id,
        timestamp: new Date(response.created_at * 1000),
        modelId: response.model,
        headers: responseHeaders,
        body: rawResponse,
      },
      providerMetadata: {
        ollama: {
          responseId: response.id,
        },
      },
      warnings,
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
      successfulResponseHandler: createEventSourceResponseHandler(
        ollamaResponsesChunkSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const self = this;

    let finishReason: LanguageModelV2FinishReason = "unknown";
    const usage: LanguageModelV2Usage = {
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

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof ollamaResponsesChunkSchema>>,
          LanguageModelV2StreamPart
        >({
          start(controller) {
            console.log("Stream started");
            controller.enqueue({ type: "stream-start", warnings });
          },

          transform(chunk, controller) {
            console.log("Chunk received", chunk);
            if (options.includeRawChunks) {
              controller.enqueue({ type: "raw", rawValue: chunk.rawValue });
            }

            // handle failed chunk parsing / validation:
            if (!chunk.success) {
              finishReason = "error";
              controller.enqueue({ type: "error", error: chunk.error });
              return;
            }

            const value = chunk.value;

            if (isResponseOutputItemAddedChunk(value)) {
              if (value.item.type === "function_call") {
                ongoingToolCalls[value.output_index] = {
                  toolName: value.item.name,
                  toolCallId: value.item.call_id,
                };

                controller.enqueue({
                  type: "tool-input-start",
                  id: value.item.call_id,
                  toolName: value.item.name,
                });
              } else if (value.item.type === "web_search_call") {
                ongoingToolCalls[value.output_index] = {
                  toolName: "web_search_preview",
                  toolCallId: value.item.id,
                };

                controller.enqueue({
                  type: "tool-input-start",
                  id: value.item.id,
                  toolName: "web_search_preview",
                });
              } else if (value.item.type === "computer_call") {
                ongoingToolCalls[value.output_index] = {
                  toolName: "computer_use",
                  toolCallId: value.item.id,
                };

                controller.enqueue({
                  type: "tool-input-start",
                  id: value.item.id,
                  toolName: "computer_use",
                });
              } else if (value.item.type === "message") {
                controller.enqueue({
                  type: "text-start",
                  id: value.item.id,
                });
              } else if (value.item.type === "reasoning") {
                controller.enqueue({
                  type: "reasoning-start",
                  id: value.item.id,
                });
              }
            } else if (isResponseOutputItemDoneChunk(value)) {
              if (value.item.type === "function_call") {
                ongoingToolCalls[value.output_index] = undefined;
                hasToolCalls = true;

                controller.enqueue({
                  type: "tool-input-end",
                  id: value.item.call_id,
                });

                controller.enqueue({
                  type: "tool-call",
                  toolCallId: value.item.call_id,
                  toolName: value.item.name,
                  input: value.item.arguments,
                });
              } else if (value.item.type === "web_search_call") {
                ongoingToolCalls[value.output_index] = undefined;
                hasToolCalls = true;

                controller.enqueue({
                  type: "tool-input-end",
                  id: value.item.id,
                });

                controller.enqueue({
                  type: "tool-call",
                  toolCallId: value.item.id,
                  toolName: "web_search_preview",
                  input: "",
                  providerExecuted: true,
                });

                controller.enqueue({
                  type: "tool-result",
                  toolCallId: value.item.id,
                  toolName: "web_search_preview",
                  result: {
                    type: "web_search_tool_result",
                    status: value.item.status || "completed",
                  },
                  providerExecuted: true,
                });
              } else if (value.item.type === "computer_call") {
                ongoingToolCalls[value.output_index] = undefined;
                hasToolCalls = true;

                controller.enqueue({
                  type: "tool-input-end",
                  id: value.item.id,
                });

                controller.enqueue({
                  type: "tool-call",
                  toolCallId: value.item.id,
                  toolName: "computer_use",
                  input: "",
                  providerExecuted: true,
                });

                controller.enqueue({
                  type: "tool-result",
                  toolCallId: value.item.id,
                  toolName: "computer_use",
                  result: {
                    type: "computer_use_tool_result",
                    status: value.item.status || "completed",
                  },
                  providerExecuted: true,
                });
              } else if (value.item.type === "message") {
                controller.enqueue({
                  type: "text-end",
                  id: value.item.id,
                });
              } else if (value.item.type === "reasoning") {
                controller.enqueue({
                  type: "reasoning-end",
                  id: value.item.id,
                });
              }
            } else if (isResponseFunctionCallArgumentsDeltaChunk(value)) {
              const toolCall = ongoingToolCalls[value.output_index];

              if (toolCall != null) {
                controller.enqueue({
                  type: "tool-input-delta",
                  id: toolCall.toolCallId,
                  delta: value.delta,
                });
              }
            } else if (isResponseCreatedChunk(value)) {
              responseId = value.response.id;
              controller.enqueue({
                type: "response-metadata",
                id: value.response.id,
                timestamp: new Date(value.response.created_at * 1000),
                modelId: value.response.model,
              });
            } else if (isTextDeltaChunk(value)) {
              controller.enqueue({
                type: "text-delta",
                id: value.item_id,
                delta: value.delta,
              });
            } else if (isResponseReasoningSummaryTextDeltaChunk(value)) {
              controller.enqueue({
                type: "reasoning-delta",
                delta: value.delta,
                id: value.item_id,
              });
            } else if (isResponseFinishedChunk(value)) {
              finishReason = mapOllamaResponseFinishReason({
                finishReason: value.response.incomplete_details?.reason,
                hasToolCalls,
              });
              usage.inputTokens = value.response.usage.input_tokens;
              usage.outputTokens = value.response.usage.output_tokens;
              usage.totalTokens =
                value.response.usage.input_tokens +
                value.response.usage.output_tokens;
              usage.reasoningTokens =
                value.response.usage.output_tokens_details?.reasoning_tokens ??
                undefined;
              usage.cachedInputTokens =
                value.response.usage.input_tokens_details?.cached_tokens ??
                undefined;
            } else if (isResponseAnnotationAddedChunk(value)) {
              controller.enqueue({
                type: "source",
                sourceType: "url",
                id: self.config.generateId?.() ?? generateId(),
                url: value.annotation.url,
                title: value.annotation.title,
              });
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
  return chunk.type === "response.output_text.delta";
}

function isResponseOutputItemDoneChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseOutputItemDoneSchema> {
  return chunk.type === "response.output_item.done";
}

function isResponseFinishedChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseFinishedChunkSchema> {
  return (
    chunk.type === "response.completed" || chunk.type === "response.incomplete"
  );
}

function isResponseCreatedChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseCreatedChunkSchema> {
  return chunk.type === "response.created";
}

function isResponseFunctionCallArgumentsDeltaChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseFunctionCallArgumentsDeltaSchema> {
  return chunk.type === "response.function_call_arguments.delta";
}

function isResponseOutputItemAddedChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseOutputItemAddedSchema> {
  return chunk.type === "response.output_item.added";
}

function isResponseAnnotationAddedChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseAnnotationAddedSchema> {
  return chunk.type === "response.output_text.annotation.added";
}

function isResponseReasoningSummaryTextDeltaChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseReasoningSummaryTextDeltaSchema> {
  return chunk.type === "response.reasoning_summary_text.delta";
}

function isResponseReasoningSummaryPartDoneChunk(
  chunk: z.infer<typeof ollamaResponsesChunkSchema>,
): chunk is z.infer<typeof responseReasoningSummaryPartDoneSchema> {
  return chunk.type === "response.reasoning_summary_part.done";
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
});

export type OllamaResponsesProviderOptions = z.infer<
  typeof ollamaResponsesProviderOptionsSchema
>;
