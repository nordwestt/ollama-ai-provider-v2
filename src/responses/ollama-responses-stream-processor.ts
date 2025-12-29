import {
  InvalidResponseDataError,
  LanguageModelV2FinishReason,
  LanguageModelV2StreamPart,
  LanguageModelV2Usage,
} from "@ai-sdk/provider";
import { generateId, ParseResult } from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import { OllamaConfig } from "../common/ollama-config";
import { getResponseMetadata } from "../common/get-response-metadata";
import { mapOllamaFinishReason } from "../adaptors/map-ollama-finish-reason";
import {
  baseOllamaResponseSchema,
  OllamaResponse,
  extractOllamaResponseObjectsFromChunk,
} from "./ollama-responses-processor";

interface StreamState {
  finishReason: LanguageModelV2FinishReason;
  usage: LanguageModelV2Usage;
  responseId: string | null;
  ongoingToolCalls: Record<number, { toolName: string; toolCallId: string } | undefined>;
  hasToolCalls: boolean;
  isFirstChunk: boolean;
  hasTextStarted: boolean;
  hasReasoningStarted: boolean;
  textEnded: boolean;
  reasoningEnded: boolean;
  textId: string;
}

export class OllamaStreamProcessor {
  private state: StreamState;

  constructor(private config: OllamaConfig) {
    this.state = this.initializeState();
  }

  createTransformStream(warnings: any[], options: any): TransformStream<
    ParseResult<z.infer<typeof baseOllamaResponseSchema>>,
    LanguageModelV2StreamPart
  > {
    return new TransformStream({
      start: (controller) => {
        controller.enqueue({ type: "stream-start", warnings });
      },

      transform: (chunk, controller) => {
        this.processChunk(chunk, controller, options);
      },

      flush: (controller) => {
        this.finalizeStream(controller);
      },
    });
  }

  private initializeState(): StreamState {
    return {
      finishReason: "unknown",
      usage: {
        inputTokens: undefined,
        outputTokens: undefined,
        totalTokens: undefined,
      },
      responseId: null,
      ongoingToolCalls: {},
      hasToolCalls: false,
      isFirstChunk: true,
      hasTextStarted: false,
      hasReasoningStarted: false,
      textEnded: false,
      reasoningEnded: false,
      textId: generateId(),
    };
  }

  private processChunk(
    chunk: ParseResult<z.infer<typeof baseOllamaResponseSchema>>,
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
    options: any,
  ) {
    if ((options as any)?.includeRawChunks) {
      controller.enqueue({ type: "raw", rawValue: (chunk as any).rawValue });
    }

    const values = extractOllamaResponseObjectsFromChunk(chunk);

    if (values.length === 0) {
      if (!chunk.success) {
        this.state.finishReason = "error";
        controller.enqueue({ type: "error", error: chunk.error });
      }
      return;
    }

    for (const value of values) {
      this.processResponseValue(value, controller);
    }
  }

  private processResponseValue(
    value: OllamaResponse,
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    // Handle error-like chunks
    if ((value as any) && typeof (value as any) === "object" && "error" in (value as any)) {
      this.state.finishReason = "error";
      controller.enqueue({ type: "error", error: (value as any).error });
      return;
    }

    if (this.state.isFirstChunk) {
      this.state.isFirstChunk = false;
      controller.enqueue({
        type: "response-metadata",
        ...getResponseMetadata(value as any),
      });
    }

    if (value.done) {
      this.handleDoneChunk(value, controller);
    }

    const delta = value?.message;
    if (delta) {
      this.processDelta(delta, controller);
    }
  }

  private handleDoneChunk(
    value: OllamaResponse,
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    this.state.finishReason = mapOllamaFinishReason(value.done_reason);
    this.state.usage = {
      inputTokens: value.prompt_eval_count || 0,
      outputTokens: value.eval_count ?? undefined,
      totalTokens: (value.prompt_eval_count ?? 0) + (value.eval_count ?? 0),
    };

    // Close any started streams
    if (this.state.hasTextStarted && !this.state.textEnded) {
      controller.enqueue({ type: "text-end", id: this.state.textId });
      this.state.textEnded = true;
    }
    if (this.state.hasReasoningStarted && !this.state.reasoningEnded) {
      controller.enqueue({ type: "reasoning-end", id: "0" });
      this.state.reasoningEnded = true;
    }
  }

  private processDelta(
    delta: OllamaResponse["message"],
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    this.processTextContent(delta, controller);
    this.processThinking(delta, controller);
    this.processToolCalls(delta, controller);
  }

  private processTextContent(
    delta: OllamaResponse["message"],
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    if (delta?.content != null) {
      if (!this.state.hasTextStarted) {
        controller.enqueue({ type: "text-start", id: this.state.textId });
        this.state.hasTextStarted = true;
      }
      controller.enqueue({
        type: "text-delta",
        id: this.state.textId,
        delta: delta.content,
      });
    }
  }

  private processThinking(
    delta: OllamaResponse["message"],
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    if (delta?.thinking) {
      if (!this.state.hasReasoningStarted) {
        controller.enqueue({ type: "reasoning-start", id: "0" });
        this.state.hasReasoningStarted = true;
      }
      controller.enqueue({
        type: "reasoning-delta",
        id: "0",
        delta: delta.thinking,
      });
    }
  }

  private processToolCalls(
    delta: OllamaResponse["message"],
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    for (const toolCall of delta.tool_calls ?? []) {
      if (toolCall.function?.name == null) {
        throw new InvalidResponseDataError({
          data: toolCall,
          message: `Expected 'function.name' to be a string.`,
        });
      }

      if (
        toolCall.function?.name != null &&
        toolCall.function?.arguments != null
      ) {
        this.emitToolCall(toolCall, controller);
      }
    }
  }

  private emitToolCall(
    toolCall: NonNullable<OllamaResponse["message"]["tool_calls"]>[0],
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    const id = toolCall.id ?? (this.config.generateId?.() ?? generateId());

    controller.enqueue({
      type: "tool-input-start",
      id: id,
      toolName: toolCall.function.name,
    });

    controller.enqueue({
      type: "tool-input-delta",
      id: id,
      delta: JSON.stringify(toolCall.function.arguments),
    });

    controller.enqueue({
      type: "tool-input-end",
      id: id,
    });

    controller.enqueue({
      type: "tool-call",
      toolCallId: id,
      toolName: toolCall.function.name,
      input: JSON.stringify(toolCall.function.arguments),
    });

    this.state.hasToolCalls = true;
  }

  private finalizeStream(
    controller: TransformStreamDefaultController<LanguageModelV2StreamPart>,
  ) {
    // Ensure any started segments are properly closed
    if (this.state.hasTextStarted && !this.state.textEnded) {
      controller.enqueue({ type: "text-end", id: this.state.textId });
    }
    if (this.state.hasReasoningStarted && !this.state.reasoningEnded) {
      controller.enqueue({ type: "reasoning-end", id: "0" });
    }

    controller.enqueue({
      type: "finish",
      finishReason: this.state.finishReason,
      usage: this.state.usage,
      providerMetadata: {
        ollama: {
          responseId: this.state.responseId,
        },
      },
    });
  }
} 
