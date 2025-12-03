import {
  LanguageModelV2FilePart,
  LanguageModelV2Prompt,
} from '@ai-sdk/provider';
import { OllamaChatPrompt } from './ollama-chat-prompt.js';

export function convertToOllamaChatMessages({
  prompt,
  systemMessageMode = 'system',
}: {
  prompt: LanguageModelV2Prompt;
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
              `Unsupported system message mode: ${_exhaustiveCheck}`
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

        const userText = content
          .filter(part => part.type === 'text')
          .map(part => part.text)
          .join('');
        const images = content
          .filter(
            part => part.type === 'file' && part.mediaType.startsWith('image/')
          )
          .map(part => (part as LanguageModelV2FilePart).data);

        messages.push({
          role: 'user',
          content: userText.length > 0 ? userText : [],
          images: images.length > 0 ? images : undefined,
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
                  arguments: part.input as object,
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
          const output = toolResponse.output;

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
            role: 'tool',
            tool_call_id: toolResponse.toolCallId,
            content: contentValue,
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
