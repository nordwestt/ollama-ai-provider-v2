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
import { prepareResponsesTools } from "./ollama-responses-prepare-tools";
import { OllamaResponsesModelId } from "../ollama-responses-settings";
import { convertToOllamaChatMessages } from "../adaptors/convert-to-ollama-chat-messages";
import { getResponseMetadata } from "../get-response-metadata";
import { mapOllamaFinishReason } from "../adaptors/map-ollama-finish-reason";

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
        modelId: this.modelId,
        timestamp: new Date(),
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
            // Normalize chunk into one or more valid response objects (handles NDJSON edge cases)
            const values = extractOllamaResponseObjectsFromChunk(chunk);

            if (values.length === 0) {
              // If we could not parse anything meaningful, propagate the original error if present
              if (!chunk.success) {
                finishReason = "error";
                controller.enqueue({ type: "error", error: chunk.error });
                return;
              }
              return;
            }

            for (const value of values) {
              // handle error-like chunks (not expected for base schema but kept for safety)
              if ((value as any) && typeof (value as any) === "object" && "error" in (value as any)) {
                finishReason = "error";
                controller.enqueue({ type: "error", error: (value as any).error });
                continue;
              }

              if (isFirstChunk) {
                isFirstChunk = false;

                controller.enqueue({
                  type: "response-metadata",
                  ...getResponseMetadata(value as any),
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

/**
 * Extracts one or more valid Ollama response objects from a stream chunk.
 * Handles both successful parsed chunks and error chunks that may contain
 * multiple JSON objects separated by newlines (NDJSON-like behavior).
 */
function extractOllamaResponseObjectsFromChunk(
  chunk: ParseResult<z.infer<typeof baseOllamaResponseSchema>>,
): Array<z.infer<typeof baseOllamaResponseSchema>> {
  if (chunk.success) {
    return [chunk.value];
  }

  const results: Array<z.infer<typeof baseOllamaResponseSchema>> = [];
  const raw = (chunk.error as any)?.text;
  if (typeof raw !== "string" || raw.length === 0) {
    return results;
  }

  const lines = raw.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === "") continue;
    try {
      const parsed = JSON.parse(trimmed);
      const validated = baseOllamaResponseSchema.safeParse(parsed);
      if (validated.success) {
        results.push(validated.data);
      }
    } catch {
      // Ignore malformed line; continue with remaining lines
    }
  }

  return results;
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
