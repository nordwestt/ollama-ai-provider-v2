import {
  InvalidResponseDataError,
  LanguageModelV1,
  LanguageModelV1CallWarning,
  LanguageModelV1FinishReason,
  LanguageModelV1LogProbs,
  LanguageModelV1ProviderMetadata,
  LanguageModelV1StreamPart,
  UnsupportedFunctionalityError,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  ParseResult,
  combineHeaders,
  createJsonStreamResponseHandler,
  createJsonResponseHandler,
  generateId,
  isParsableJson,
  postJsonToApi,
} from '@ai-sdk/provider-utils';
import { z } from 'zod';
import { convertToOllamaChatMessages } from './convert-to-ollama-chat-messages';
import { mapOllamaChatLogProbsOutput } from './map-ollama-chat-logprobs';
import { mapOllamaFinishReason } from './map-ollama-finish-reason';
import { OllamaChatModelId, OllamaChatSettings } from './ollama-chat-settings';
import {
  ollamaErrorDataSchema,
  ollamaFailedResponseHandler,
} from './ollama-error';
import { getResponseMetadata } from './get-response-metadata';
import { prepareTools } from './ollama-prepare-tools';

type OllamaChatConfig = {
  provider: string;
  compatibility: 'strict' | 'compatible';
  headers: () => Record<string, string | undefined>;
  url: (options: { modelId: string; path: string }) => string;
  fetch?: FetchFunction;
};

export class OllamaChatLanguageModel implements LanguageModelV1 {
  readonly specificationVersion = 'v1';

  readonly modelId: OllamaChatModelId;
  readonly settings: OllamaChatSettings;

  private readonly config: OllamaChatConfig;

  constructor(
    modelId: OllamaChatModelId,
    settings: OllamaChatSettings,
    config: OllamaChatConfig,
  ) {
    this.modelId = modelId;
    this.settings = settings;
    this.config = config;
  }

  get supportsStructuredOutputs(): boolean {
    // enable structured outputs for reasoning models by default:
    // TODO in the next major version, remove this and always use json mode for models
    // that support structured outputs (blacklist other models)
    return this.settings.structuredOutputs ?? isReasoningModel(this.modelId);
  }

  get defaultObjectGenerationMode() {
    // audio models don't support structured outputs:
    if (isAudioModel(this.modelId)) {
      return 'tool';
    }

    return this.supportsStructuredOutputs ? 'json' : 'tool';
  }

  get provider(): string {
    return this.config.provider;
  }

  get supportsImageUrls(): boolean {
    // image urls can be sent if downloadImages is disabled (default):
    return !this.settings.downloadImages;
  }

  private getArgs({
    mode,
    prompt,
    maxTokens,
    temperature,
    topP,
    topK,
    frequencyPenalty,
    presencePenalty,
    stopSequences,
    responseFormat,
    seed,
    providerMetadata,
  }: Parameters<LanguageModelV1['doGenerate']>[0]) {
    const type = mode.type;

    const warnings: LanguageModelV1CallWarning[] = [];

    if (topK != null) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'topK',
      });
    }

    if (
      responseFormat?.type === 'json' &&
      responseFormat.schema != null &&
      !this.supportsStructuredOutputs
    ) {
      warnings.push({
        type: 'unsupported-setting',
        setting: 'responseFormat',
        details:
          'JSON response format schema is only supported with structuredOutputs',
      });
    }

    if (
      getSystemMessageMode(this.modelId) === 'remove' &&
      prompt.some(message => message.role === 'system')
    ) {
      warnings.push({
        type: 'other',
        message: 'system messages are removed for this model',
      });
    }

    const baseArgs = {
      // model id:
      model: this.modelId,

      // model specific settings:
      logit_bias: this.settings.logitBias,
      logprobs:
        this.settings.logprobs === true ||
        typeof this.settings.logprobs === 'number'
          ? true
          : undefined,
      top_logprobs:
        typeof this.settings.logprobs === 'number'
          ? this.settings.logprobs
          : typeof this.settings.logprobs === 'boolean'
          ? this.settings.logprobs
            ? 0
            : undefined
          : undefined,
      user: this.settings.user,
      parallel_tool_calls: this.settings.parallelToolCalls,

      // standardized settings:
      max_tokens: maxTokens,
      temperature,
      top_p: topP,
      frequency_penalty: frequencyPenalty,
      presence_penalty: presencePenalty,
      response_format:
        responseFormat?.type === 'json'
          ? this.supportsStructuredOutputs && responseFormat.schema != null
            ? {
                type: 'json_schema',
                json_schema: {
                  schema: responseFormat.schema,
                  strict: true,
                  name: responseFormat.name ?? 'response',
                  description: responseFormat.description,
                },
              }
            : { type: 'json_object' }
          : undefined,
      stop: stopSequences,
      seed,

      // ollama specific settings:
      // TODO remove in next major version; we auto-map maxTokens now
      max_completion_tokens: providerMetadata?.ollama?.maxCompletionTokens,
      store: providerMetadata?.ollama?.store,
      metadata: providerMetadata?.ollama?.metadata,
      prediction: providerMetadata?.ollama?.prediction,
      reasoning_effort:
        providerMetadata?.ollama?.reasoningEffort ??
        this.settings.reasoningEffort,
      think: providerMetadata?.ollama?.think ?? this.settings.think,


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
          type: 'unsupported-setting',
          setting: 'temperature',
          details: 'temperature is not supported for reasoning models',
        });
      }
      if (baseArgs.top_p != null) {
        baseArgs.top_p = undefined;
        warnings.push({
          type: 'unsupported-setting',
          setting: 'topP',
          details: 'topP is not supported for reasoning models',
        });
      }
      if (baseArgs.frequency_penalty != null) {
        baseArgs.frequency_penalty = undefined;
        warnings.push({
          type: 'unsupported-setting',
          setting: 'frequencyPenalty',
          details: 'frequencyPenalty is not supported for reasoning models',
        });
      }
      if (baseArgs.presence_penalty != null) {
        baseArgs.presence_penalty = undefined;
        warnings.push({
          type: 'unsupported-setting',
          setting: 'presencePenalty',
          details: 'presencePenalty is not supported for reasoning models',
        });
      }
      if (baseArgs.logit_bias != null) {
        baseArgs.logit_bias = undefined;
        warnings.push({
          type: 'other',
          message: 'logitBias is not supported for reasoning models',
        });
      }
      if (baseArgs.logprobs != null) {
        baseArgs.logprobs = undefined;
        warnings.push({
          type: 'other',
          message: 'logprobs is not supported for reasoning models',
        });
      }
      if (baseArgs.top_logprobs != null) {
        baseArgs.top_logprobs = undefined;
        warnings.push({
          type: 'other',
          message: 'topLogprobs is not supported for reasoning models',
        });
      }

      // reasoning models use max_completion_tokens instead of max_tokens:
      if (baseArgs.max_tokens != null) {
        if (baseArgs.max_completion_tokens == null) {
          baseArgs.max_completion_tokens = baseArgs.max_tokens;
        }
        baseArgs.max_tokens = undefined;
      }
    }

    switch (type) {
      case 'regular': {
        const { tools, tool_choice, functions, function_call, toolWarnings } =
          prepareTools({
            mode,
            structuredOutputs: this.supportsStructuredOutputs,
          });

        return {
          args: {
            ...baseArgs,
            tools,
            tool_choice,
            functions,
            function_call,
          },
          warnings: [...warnings, ...toolWarnings],
        };
      }

      case 'object-json': {
        return {
          args: {
            ...baseArgs,
            response_format:
              this.supportsStructuredOutputs && mode.schema != null
                ? {
                    type: 'json_schema',
                    json_schema: {
                      schema: mode.schema,
                      strict: true,
                      name: mode.name ?? 'response',
                      description: mode.description,
                    },
                  }
                : { type: 'json_object' },
          },
          warnings,
        };
      }

      case 'object-tool': {
        return {
          args: {
                ...baseArgs,
                tool_choice: {
                  type: 'function',
                  function: { name: mode.tool.name },
                },
                tools: [
                  {
                    type: 'function',
                    function: {
                      name: mode.tool.name,
                      description: mode.tool.description,
                      parameters: mode.tool.parameters,
                      strict: this.supportsStructuredOutputs ? true : undefined,
                    },
                  },
                ],
              },
          warnings,
        };
      }

      default: {
        const _exhaustiveCheck: never = type;
        throw new Error(`Unsupported type: ${_exhaustiveCheck}`);
      }
    }
  }

  async doGenerate(
    options: Parameters<LanguageModelV1['doGenerate']>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV1['doGenerate']>>> {
    const { args: body, warnings } = this.getArgs(options);

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: '/chat',
        modelId: this.modelId,
      }),
      headers: combineHeaders(this.config.headers(), options.headers),
      body: {...body, stream:false},
      failedResponseHandler: ollamaFailedResponseHandler,
      successfulResponseHandler: createJsonResponseHandler(
        baseOllamaResponseSchema,
      ),
      abortSignal: options.abortSignal,
      fetch: this.config.fetch,
    });

    const { messages: rawPrompt, ...rawSettings } = body;

    // provider metadata:
    const providerMetadata: LanguageModelV1ProviderMetadata = { ollama: {} };


    return {
      text: response.message.content ?? undefined,
      reasoning: response.message.thinking,
      toolCalls: response.message.tool_calls?.map(toolCall => ({
              toolCallType: 'function',
              toolCallId: toolCall.id ?? generateId(),
              toolName: toolCall.function.name,
              args: JSON.stringify(toolCall.function.arguments),
            })),
      finishReason: mapOllamaFinishReason(response.done_reason),
      usage: {
        promptTokens: response.prompt_eval_count ?? NaN,
        completionTokens: response.eval_count ?? NaN,
      },
      rawCall: { rawPrompt, rawSettings },
      rawResponse: { headers: responseHeaders },
      request: { body: JSON.stringify(body) },
      response: getResponseMetadata(response),
      warnings,
      logprobs: undefined,
      providerMetadata,
    };
  }

  async doStream(
    options: Parameters<LanguageModelV1['doStream']>[0],
  ): Promise<Awaited<ReturnType<LanguageModelV1['doStream']>>> {
    if (
      this.settings.simulateStreaming ??
      isStreamingSimulatedByDefault(this.modelId)
    ) {
      const result = await this.doGenerate(options);

      const simulatedStream = new ReadableStream<LanguageModelV1StreamPart>({
        start(controller) {
          controller.enqueue({ type: 'response-metadata', ...result.response });
          if (result.text) {
            controller.enqueue({
              type: 'text-delta',
              textDelta: result.text,
            });
          }
          if (result.toolCalls) {
            for (const toolCall of result.toolCalls) {
              controller.enqueue({
                type: 'tool-call-delta',
                toolCallType: 'function',
                toolCallId: toolCall.toolCallId,
                toolName: toolCall.toolName,
                argsTextDelta: toolCall.args,
              });

              controller.enqueue({
                type: 'tool-call',
                ...toolCall,
              });
            }
          }
          controller.enqueue({
            type: 'finish',
            finishReason: result.finishReason,
            usage: result.usage,
            logprobs: result.logprobs,
            providerMetadata: result.providerMetadata,
          });
          controller.close();
        },
      });
      return {
        stream: simulatedStream,
        rawCall: result.rawCall,
        rawResponse: result.rawResponse,
        warnings: result.warnings,
      };
    }

    const { args, warnings } = this.getArgs(options);

    const body = {
      ...args,
      stream: true,

      // only include stream_options when in strict compatibility mode:
      stream_options:
        this.config.compatibility === 'strict'
          ? { include_usage: true }
          : undefined,
    };

    const { responseHeaders, value: response } = await postJsonToApi({
      url: this.config.url({
        path: '/chat',
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

    let finishReason: LanguageModelV1FinishReason = 'unknown';
    let usage: {
      promptTokens: number | undefined;
      completionTokens: number | undefined;
    } = {
      promptTokens: undefined,
      completionTokens: undefined,
    };
    let isFirstChunk = true;

    return {
      stream: response.pipeThrough(
        new TransformStream<
          ParseResult<z.infer<typeof baseOllamaResponseSchema>>,
          LanguageModelV1StreamPart
        >({
          flush(controller) {
            controller.enqueue({
              type: 'finish',
              finishReason,
              usage: {
                promptTokens: usage.promptTokens ?? NaN,
                completionTokens: usage.completionTokens ?? NaN,
              },
            });
          },
          transform(chunk, controller) {
            // handle failed chunk parsing / validation:
            if (!chunk.success) {
              
              try{
                const text = (chunk.error as any).text as string;
                const lines = text.split('\n');
                lines.forEach(line => {
                  if(line.trim()==='') return;
                  const parsed = JSON.parse(line);
                  controller.enqueue({
                    type: 'text-delta',
                    textDelta: parsed.message.content,
                  });
                  controller.enqueue({
                    type: 'reasoning',
                    textDelta: parsed.message.thinking
                  });
                });
                return;
              }
              catch(e){
                console.error('ccchunk error', e);
              }
              console.error('chunk error', Object.keys((chunk.error as any).text));
              finishReason = 'error';
              controller.enqueue({ type: 'error', error: chunk.error });
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


            if(value.done){
              finishReason = mapOllamaFinishReason(value.done_reason);
              usage = {
                promptTokens: value.prompt_eval_count || 0,
                completionTokens: value.eval_count ?? undefined,
              };
            }
            const delta = value?.message;

            if (delta?.content != null) {
              controller.enqueue({
                type: 'text-delta',
                textDelta: delta.content,
              });
            }

            if(delta?.thinking){
              controller.enqueue({
                type: 'reasoning',
                textDelta: delta.thinking
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
                toolCall.function?.arguments != null && Object.keys(toolCall.function.arguments).length > 0
              ) {

                const id = generateId();

                controller.enqueue({
                  type: 'tool-call-delta',
                  toolCallType: 'function',
                  toolCallId: id,
                  toolName: toolCall.function.name,
                  argsTextDelta: JSON.stringify(toolCall.function.arguments),
                });

                controller.enqueue({
                  type: 'tool-call',
                  toolCallType: 'function',
                  toolCallId: id,
                  toolName: toolCall.function.name,
                  args: JSON.stringify(toolCall.function.arguments),
                });
              }
            }
          },
        }),
      ),
      rawCall: { rawPrompt, rawSettings },
      rawResponse: { headers: responseHeaders },
      request: { body: JSON.stringify(body) },
      warnings,
    };
  }
}

const ollamaTokenUsageSchema = z
  .object({
    prompt_tokens: z.number().nullish(),
    completion_tokens: z.number().nullish(),
    prompt_tokens_details: z
      .object({
        cached_tokens: z.number().nullish(),
      })
      .nullish(),
    completion_tokens_details: z
      .object({
        reasoning_tokens: z.number().nullish(),
        accepted_prediction_tokens: z.number().nullish(),
        rejected_prediction_tokens: z.number().nullish(),
      })
      .nullish(),
  })
  .nullish();

const baseOllamaResponseSchema = z.object({
  model: z.string(),
  created_at: z.string(),
  done: z.boolean(),
  message: z.object({
    content: z.string(),
    role: z.string(),
    thinking: z.string().optional(),
    tool_calls: z.array(z.object({
      function: z.object({
        name: z.string(),
        arguments: z.record(z.any())
      }),
      id: z.string().optional(),
    })).optional().nullable()
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
    modelId === 'o1' ||
    modelId.startsWith('o1-') ||
    modelId === 'o3' ||
    modelId.startsWith('o3-')
  );
}

function isAudioModel(modelId: string) {
  return modelId.startsWith('gpt-4o-audio-preview');
}

function getSystemMessageMode(modelId: string) {
  if (!isReasoningModel(modelId)) {
    return 'system';
  }

  return (
    reasoningModels[modelId as keyof typeof reasoningModels]
      ?.systemMessageMode ?? 'developer'
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
  'o1-mini': {
    systemMessageMode: 'remove',
    simulateStreamingByDefault: false,
  },
  'o1-mini-2024-09-12': {
    systemMessageMode: 'remove',
    simulateStreamingByDefault: false,
  },
  'o1-preview': {
    systemMessageMode: 'remove',
    simulateStreamingByDefault: false,
  },
  'o1-preview-2024-09-12': {
    systemMessageMode: 'remove',
    simulateStreamingByDefault: false,
  },
} as const;
