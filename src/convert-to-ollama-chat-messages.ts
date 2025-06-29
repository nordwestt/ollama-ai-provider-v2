import {
  LanguageModelV1Prompt,
  UnsupportedFunctionalityError,
} from '@ai-sdk/provider';
import { convertUint8ArrayToBase64 } from '@ai-sdk/provider-utils';
import { OllamaChatPrompt } from './ollama-chat-prompt';

export function convertToOllamaChatMessages({
  prompt,
  systemMessageMode = 'system',
}: {
  prompt: LanguageModelV1Prompt;
  systemMessageMode?: 'system' | 'developer' | 'remove';
}): OllamaChatPrompt {
  const messages: OllamaChatPrompt = [];

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
            break;
          }
          default: {
            const _exhaustiveCheck: never = systemMessageMode;
            throw new Error(
              `Unsupported system message mode: ${_exhaustiveCheck}`,
            );
          }
        }
        break;
      }

      case 'user': {
        if (content.length === 1 && content[0].type === 'text') {
          messages.push({ role: 'user', content: content[0].text });
          break;
        }

        messages.push({
          role: 'user',
          content: content.map(part => {
            switch (part.type) {
              case 'text': {
                return { type: 'text', text: part.text };
              }
              case 'image': {
                return {
                  type: 'image_url',
                  image_url: {
                    url:
                      part.image instanceof URL
                        ? part.image.toString()
                        : `data:${
                            part.mimeType ?? 'image/jpeg'
                          };base64,${convertUint8ArrayToBase64(part.image)}`,

                    // Ollama specific extension: image detail
                    detail: part.providerMetadata?.ollama?.imageDetail,
                  },
                };
              }
              case 'file': {
                if (part.data instanceof URL) {
                  throw new UnsupportedFunctionalityError({
                    functionality:
                      "'File content parts with URL data' functionality not supported.",
                  });
                }

                switch (part.mimeType) {
                  case 'audio/wav': {
                    return {
                      type: 'input_audio',
                      input_audio: { data: part.data, format: 'wav' },
                    };
                  }
                  case 'audio/mp3':
                  case 'audio/mpeg': {
                    return {
                      type: 'input_audio',
                      input_audio: { data: part.data, format: 'mp3' },
                    };
                  }

                  default: {
                    throw new UnsupportedFunctionalityError({
                      functionality: `File content part type ${part.mimeType} in user messages`,
                    });
                  }
                }
              }
            }
          }),
        });

        break;
      }

      case 'assistant': {
        let text = '';
        let thinking = '';
        const toolCalls: Array<{
          id: string;
          type: 'function';
          function: { name: string; arguments: object };
        }> = [];

        for (const part of content) {
          switch (part.type) {
            case 'text': {
              text += part.text;
              break;
            }
            case 'tool-call': {
              toolCalls.push({
                id: part.toolCallId,
                type: 'function',
                function: {
                  name: part.toolName,
                  arguments: part.args as object,
                },
              });
              break;
            }
            case 'reasoning': {
              thinking += part.text;
              break;
            } 
            default: {
              throw new Error(`Unsupported part: ${part}`);
            }
          }
        }
        
        messages.push({
          role: 'assistant',
          content: text,
          ...(thinking && { thinking }),
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
        });

        break;
      }

      case 'tool': {
        for (const toolResponse of content) {
          messages.push({
            role: 'tool',
            tool_call_id: toolResponse.toolCallId,
            content: JSON.stringify(toolResponse.result),
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

  return messages;
}
