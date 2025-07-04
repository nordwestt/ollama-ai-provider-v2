import {
    LanguageModelV2CallOptions,
    LanguageModelV2CallWarning,
    UnsupportedFunctionalityError,
  } from '@ai-sdk/provider';
  import { OllamaResponsesTool } from './ollama-responses-api-types';
  
  export function prepareResponsesTools({
    tools,
    toolChoice,
    strictJsonSchema,
  }: {
    tools: LanguageModelV2CallOptions['tools'];
    toolChoice?: LanguageModelV2CallOptions['toolChoice'];
    strictJsonSchema: boolean;
  }): {
    tools?: Array<OllamaResponsesTool>;
    toolChoice?:
      | 'auto'
      | 'none'
      | 'required'
      | { type: 'web_search_preview' }
      | { type: 'function'; name: string };
    toolWarnings: LanguageModelV2CallWarning[];
  } {
    // when the tools array is empty, change it to undefined to prevent errors:
    tools = tools?.length ? tools : undefined;
  
    const toolWarnings: LanguageModelV2CallWarning[] = [];
  
    if (tools == null) {
      return { tools: undefined, toolChoice: undefined, toolWarnings };
    }
  
    const ollamaTools: Array<OllamaResponsesTool> = [];
  
    for (const tool of tools) {
      switch (tool.type) {
        case 'function':
          ollamaTools.push({
            type: 'function',
            name: tool.name,
            description: tool.description,
            parameters: tool.inputSchema,
            strict: strictJsonSchema,
          });
          break;
        case 'provider-defined':
          switch (tool.id) {
            case 'ollama.web_search_preview':
              ollamaTools.push({
                type: 'web_search_preview',
                search_context_size: tool.args.searchContextSize as
                  | 'low'
                  | 'medium'
                  | 'high',
                user_location: tool.args.userLocation as {
                  type: 'approximate';
                  city: string;
                  region: string;
                },
              });
              break;
            default:
              toolWarnings.push({ type: 'unsupported-tool', tool });
              break;
          }
          break;
        default:
          toolWarnings.push({ type: 'unsupported-tool', tool });
          break;
      }
    }
  
    if (toolChoice == null) {
      return { tools: ollamaTools, toolChoice: undefined, toolWarnings };
    }
  
    const type = toolChoice.type;
  
    switch (type) {
      case 'auto':
      case 'none':
      case 'required':
        return { tools: ollamaTools, toolChoice: type, toolWarnings };
      case 'tool':
        return {
          tools: ollamaTools,
          toolChoice:
            toolChoice.toolName === 'web_search_preview'
              ? { type: 'web_search_preview' }
              : { type: 'function', name: toolChoice.toolName },
          toolWarnings,
        };
      default: {
        const _exhaustiveCheck: never = type;
        throw new UnsupportedFunctionalityError({
          functionality: `tool choice type: ${_exhaustiveCheck}`,
        });
      }
    }
  }
  