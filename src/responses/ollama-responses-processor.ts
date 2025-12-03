import {
  LanguageModelV3Content,
  LanguageModelV3FinishReason,
  LanguageModelV3Usage,
  SharedV3ProviderMetadata,
} from '@ai-sdk/provider';
import { generateId } from '@ai-sdk/provider-utils';
import { z } from 'zod/v4';
import { mapOllamaFinishReason } from '../adaptors/map-ollama-finish-reason.js';
import { OllamaConfig } from '../common/ollama-config.js';

export const baseOllamaResponseSchema = z.object({
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
        })
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

export type OllamaResponse = z.infer<typeof baseOllamaResponseSchema>;

export class OllamaResponseProcessor {
  constructor(private config: OllamaConfig) {}

  processGenerateResponse(response: OllamaResponse): {
    content: LanguageModelV3Content[];
    finishReason: LanguageModelV3FinishReason;
    usage: LanguageModelV3Usage;
    providerMetadata: SharedV3ProviderMetadata;
  } {
    const content = this.extractContent(response);
    const finishReason = mapOllamaFinishReason(response.done_reason);
    const usage = this.extractUsage(response);
    const providerMetadata: SharedV3ProviderMetadata = { ollama: {} };

    return {
      content,
      finishReason,
      usage,
      providerMetadata,
    };
  }

  private extractContent(response: OllamaResponse): LanguageModelV3Content[] {
    const content: LanguageModelV3Content[] = [];

    // Add text content
    const text = response.message.content;
    if (text != null && text.length > 0) {
      content.push({
        type: 'text',
        text,
      });
    }

    // Add thinking content
    const thinking = response.message.thinking;
    if (thinking != null && thinking.length > 0) {
      content.push({
        type: 'reasoning',
        text: thinking,
      });
    }

    // Add tool calls
    for (const toolCall of response.message.tool_calls ?? []) {
      content.push({
        type: 'tool-call' as const,
        toolCallId: toolCall.id ?? this.config.generateId?.() ?? generateId(),
        toolName: toolCall.function.name,
        input: JSON.stringify(toolCall.function.arguments),
      });
    }

    return content;
  }

  private extractUsage(response: OllamaResponse): LanguageModelV3Usage {
    return {
      inputTokens: response.prompt_eval_count ?? undefined,
      outputTokens: response.eval_count ?? undefined,
      totalTokens:
        (response.prompt_eval_count ?? 0) + (response.eval_count ?? 0),
      reasoningTokens: undefined, // Ollama doesn't provide separate reasoning tokens
      cachedInputTokens: undefined,
    };
  }
}

/**
 * Extracts one or more valid Ollama response objects from a stream chunk.
 * Handles both successful parsed chunks and error chunks that may contain
 * multiple JSON objects separated by newlines (NDJSON-like behavior).
 */
export function extractOllamaResponseObjectsFromChunk(
  chunk: any
): OllamaResponse[] {
  if (chunk.success) {
    return [chunk.value];
  }

  const results: OllamaResponse[] = [];
  const raw = (chunk.error as any)?.text;
  if (typeof raw !== 'string' || raw.length === 0) {
    return results;
  }

  const lines = raw.split(/\r?\n/);
  for (const line of lines) {
    const trimmed = line.trim();
    if (trimmed === '') continue;
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
