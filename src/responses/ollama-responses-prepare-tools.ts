import {
  LanguageModelV2CallOptions,
  LanguageModelV2CallWarning,
  UnsupportedFunctionalityError,
} from "@ai-sdk/provider";
import { OllamaResponsesTool } from "./ollama-responses-api-types";

export function prepareResponsesTools({
  tools,
  toolChoice,
}: {
  tools: LanguageModelV2CallOptions["tools"];
  toolChoice?: LanguageModelV2CallOptions["toolChoice"];
}): {
  tools?: Array<OllamaResponsesTool>;
  toolChoice?:
    | "auto"
    | "none"
    | "required"
    | { type: "web_search_preview" }
    | { type: "function"; name: string };
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
      case "function": {
        // Ensure parameters is always a non-null object (even if empty)
        let parameters = tool.inputSchema;
        if(!parameters){
          parameters = {
            type: "object",
            properties: {},
            required: [],
          };
        }
        else if (
          parameters &&
          typeof parameters === "object" &&
          parameters.type === "object" &&
          parameters.properties &&
          Object.keys(parameters.properties).length === 0
        ) {
          // Defensive: ensure required/optional fields are handled for empty schemas
          parameters = {
            ...parameters,
            properties: {},
            required: [],
          };
        } 
        
        ollamaTools.push({
          type: "function",
          function: {
            name: tool.name,
            description: tool.description,
            parameters,
          },
        });
        break;
      }
      default:
        toolWarnings.push({ type: "unsupported-tool", tool });
        break;
    }
  }

  if (toolChoice == null) {
    return { tools: ollamaTools, toolChoice: undefined, toolWarnings };
  }

  const type = toolChoice.type;

  switch (type) {
    case "auto":
    case "none":
    case "required":
      return { tools: ollamaTools, toolChoice: type, toolWarnings };
    case "tool":
      return {
        tools: ollamaTools,
        toolChoice:
          toolChoice.toolName == "web_search_preview"
            ? { type: "web_search_preview" }
            : { type: "function", name: toolChoice.toolName },
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
