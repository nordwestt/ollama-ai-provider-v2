import {
  SharedV3Warning,
  LanguageModelV3Prompt,
  UnsupportedFunctionalityError,
} from '@ai-sdk/provider';
import { OllamaResponsesPrompt } from './ollama-responses-api-types.js';

export function convertToOllamaResponsesMessages({
  prompt,
  systemMessageMode,
}: {
  prompt: LanguageModelV3Prompt;
  systemMessageMode: 'system' | 'developer' | 'remove';
}): {
  messages: OllamaResponsesPrompt;
  warnings: Array<SharedV3Warning>;
} {
  const messages: OllamaResponsesPrompt = [];
  const warnings: Array<SharedV3Warning> = [];

  for (const { role, content } of prompt) {
    switch (role) {
      case 'system': {
        switch (systemMessageMode) {
          case 'system': {
            messages.push({ role: 'system', content });
            break;
          }
          case 'developer': {
            messages.push({ role: 'developer', content });
            break;
          }
          case 'remove': {
            warnings.push({
              type: 'other',
              message: 'system messages are removed for this model',
            });
            break;
          }
          default: {
            const _exhaustiveCheck: never = systemMessageMode;
            throw new Error(
              `Unsupported system message mode: ${_exhaustiveCheck}`
            );
          }
        }
        break;
      }

      case 'user': {
        messages.push({
          role: 'user',
          content: content.map((part, index) => {
            switch (part.type) {
              case 'text': {
                return { type: 'input_text', text: part.text };
              }
              case 'file': {
                if (part.mediaType.startsWith('image/')) {
                  const mediaType =
                    part.mediaType === 'image/*'
                      ? 'image/jpeg'
                      : part.mediaType;

                  return {
                    type: 'input_image',
                    image_url:
                      part.data instanceof URL
                        ? part.data.toString()
                        : `data:${mediaType};base64,${part.data}`,

                    // Ollama specific extension: image detail
                    detail: part.providerOptions?.ollama?.imageDetail,
                  };
                } else if (part.mediaType === 'application/pdf') {
                  if (part.data instanceof URL) {
                    // The AI SDK automatically downloads files for user file parts with URLs
                    throw new UnsupportedFunctionalityError({
                      functionality: 'PDF file parts with URLs',
                    });
                  }

                  return {
                    type: 'input_file',
                    filename: part.filename ?? `part-${index}.pdf`,
                    file_data: `data:application/pdf;base64,${part.data}`,
                  };
                } else {
                  throw new UnsupportedFunctionalityError({
                    functionality: `file part media type ${part.mediaType}`,
                  });
                }
              }
            }
          }),
        });

        break;
      }

      case 'assistant': {
        for (const part of content) {
          switch (part.type) {
            case 'text': {
              messages.push({
                role: 'assistant',
                content: [{ type: 'output_text', text: part.text }],
              });
              break;
            }
            case 'tool-call': {
              if (part.providerExecuted) {
                break;
              }

              messages.push({
                type: 'function_call',
                call_id: part.toolCallId,
                name: part.toolName,
                arguments: JSON.stringify(part.input),
              });
              break;
            }

            case 'tool-result': {
              warnings.push({
                type: 'other',
                message: `tool result parts in assistant messages are not supported for Ollama responses`,
              });
              break;
            }
          }
        }

        break;
      }

      case 'tool': {
        for (const part of content) {
          const output = part.output;

          let contentValue: string;
          switch (output.type) {
            case 'text':
            case 'error-text':
              contentValue = output.value;
              break;
            case 'content':
            case 'json':
            case 'error-json':
              contentValue = JSON.stringify(output.value);
              break;
          }

          messages.push({
            type: 'function_call_output',
            call_id: part.toolCallId,
            output: contentValue,
          });
        }

        break;
      }

      default: {
        const _exhaustiveCheck: never = role;
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`);
      }
    }
  }

  return { messages, warnings };
}
