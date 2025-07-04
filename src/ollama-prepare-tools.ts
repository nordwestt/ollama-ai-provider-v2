import {
  JSONSchema7,
  LanguageModelV2,
  LanguageModelV2CallOptions,
  LanguageModelV2CallWarning,
  UnsupportedFunctionalityError,
} from '@ai-sdk/provider';
import { OllamaToolChoice, OllamaTools } from './ollama-types';

export function prepareTools({
  tools,
  toolChoice,
  structuredOutputs,
  strictJsonSchema,
}: {
  tools: LanguageModelV2CallOptions['tools'];
  toolChoice?: LanguageModelV2CallOptions['toolChoice'];
  structuredOutputs: boolean;
  strictJsonSchema: boolean;
}): {
  tools?: OllamaTools;
  toolChoice?: OllamaToolChoice;
  toolWarnings: Array<LanguageModelV2CallWarning>;
} {
  // when the tools array is empty, change it to undefined to prevent errors:
  tools = tools?.length ? tools : undefined;

  const toolWarnings: LanguageModelV2CallWarning[] = [];

  if (tools == null) {
    return { tools: undefined, toolChoice: undefined, toolWarnings };
  }

  const ollamaTools: OllamaTools = [];

  for (const tool of tools) {
    if(tool.type === 'function'){
      ollamaTools.push({
        type: 'function',
        function: {
          name: tool.name,
          description: tool.description,
          parameters: tool.inputSchema,
          strict: structuredOutputs ? strictJsonSchema : undefined,
        },
      });
    }
    else {
      toolWarnings.push({ type: 'unsupported-tool', tool });
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
        toolChoice: {
          type: 'function',
          function: {
            name: toolChoice.toolName,
          },
        },
        toolWarnings,
      };
    default: {
      const _exhaustiveCheck: never = type;
      throw new UnsupportedFunctionalityError({
        functionality: `Unsupported tool choice type: ${_exhaustiveCheck}`,
      });
    }
  }
}
