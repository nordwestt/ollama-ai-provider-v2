import { z } from 'zod';
import { extractResponseHeaders, ResponseHandler } from '@ai-sdk/provider-utils';

/**
 * Creates a response handler for NDJSON (Newline-delimited JSON) streams.
 * This replaces the removed createJsonStreamResponseHandler from @ai-sdk/provider-utils.
 */
export function createNdjsonStreamResponseHandler<T>(
  schema: z.ZodSchema<T>
): ResponseHandler<ReadableStream<T>> {
  return async ({ response }) => {
    const responseHeaders = extractResponseHeaders(response);

    if (response.body == null) {
      throw new Error('Response body is null');
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    const stream = new ReadableStream<T>({
      async pull(controller) {
        while (true) {
          const { done, value } = await reader.read();

          if (done) {
            // Process any remaining data in buffer
            if (buffer.trim()) {
              try {
                const parsed = JSON.parse(buffer.trim());
                const validated = schema.parse(parsed);
                controller.enqueue(validated);
              } catch {
                // Ignore parse errors for incomplete data at end
              }
            }
            controller.close();
            return;
          }

          buffer += decoder.decode(value, { stream: true });
          const lines = buffer.split('\n');
          buffer = lines.pop() || '';

          for (const line of lines) {
            const trimmedLine = line.trim();
            if (trimmedLine) {
              try {
                const parsed = JSON.parse(trimmedLine);
                const validated = schema.parse(parsed);
                controller.enqueue(validated);
              } catch (error) {
                // Skip invalid JSON lines
                console.warn('Failed to parse NDJSON line:', error);
              }
            }
          }
        }
      },
      cancel() {
        reader.cancel();
      },
    });

    return {
      responseHeaders,
      value: stream,
    };
  };
}
