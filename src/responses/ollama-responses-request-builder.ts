import {
  LanguageModelV2CallWarning,
  LanguageModelV2,
} from "@ai-sdk/provider";
import { parseProviderOptions } from "@ai-sdk/provider-utils";
import { z } from "zod/v4";
import { convertToOllamaResponsesMessages } from "./convert-to-ollama-responses-messages";
import { convertToOllamaChatMessages } from "../adaptors/convert-to-ollama-chat-messages";
import { prepareResponsesTools } from "./ollama-responses-prepare-tools";
import { OllamaChatModelId, ollamaProviderOptions } from "../ollama-chat-settings";


export type OllamaResponsesProviderOptions = z.infer<
  typeof ollamaProviderOptions
>;

interface RequestBuilderOptions {
  modelId: OllamaChatModelId;
  maxOutputTokens?: number;
  temperature?: number;
  stopSequences?: string[];
  topP?: number;
  topK?: number;
  presencePenalty?: number;
  frequencyPenalty?: number;
  seed?: number;
  prompt: any;
  providerOptions?: any;
  tools?: any;
  toolChoice?: any;
  responseFormat?: any;
}

interface RequestBuilderResult {
  args: {
    model: OllamaChatModelId;
    messages: any;
    temperature?: number;
    top_p?: number;
    max_output_tokens?: number;
    format?: any;
    user?: string;
    think?: boolean;
    tools?: any;
    tool_choice?: any;
  };
  warnings: LanguageModelV2CallWarning[];
}

export class OllamaRequestBuilder {
  async buildRequest({
    modelId,
    maxOutputTokens,
    temperature,
    stopSequences,
    topP,
    topK,
    presencePenalty,
    frequencyPenalty,
    seed,
    prompt,
    providerOptions,
    tools,
    toolChoice,
    responseFormat,
  }: RequestBuilderOptions): Promise<RequestBuilderResult> {
    const warnings = this.collectUnsupportedSettingsWarnings({
      topK,
      seed,
      presencePenalty,
      frequencyPenalty,
      stopSequences,
    });

    const { messages, warnings: messageWarnings } =
      convertToOllamaResponsesMessages({
        prompt,
        systemMessageMode: "system",
      });

    warnings.push(...messageWarnings);

    const ollamaOptions = await this.parseProviderOptions(providerOptions);

    const baseArgs = this.buildBaseArgs({
      modelId,
      prompt,
      temperature,
      topP,
      maxOutputTokens,
      responseFormat,
      ollamaOptions,
    });

    const { tools: ollamaTools, toolChoice: ollamaToolChoice, toolWarnings } =
      prepareResponsesTools({
        tools,
        toolChoice,
      });

    return {
      args: {
        ...baseArgs,
        tools: ollamaTools,
        tool_choice: ollamaToolChoice,
      },
      warnings: [...warnings, ...toolWarnings],
    };
  }

  private collectUnsupportedSettingsWarnings({
    topK,
    seed,
    presencePenalty,
    frequencyPenalty,
    stopSequences,
  }: {
    topK?: number;
    seed?: number;
    presencePenalty?: number;
    frequencyPenalty?: number;
    stopSequences?: string[];
  }): LanguageModelV2CallWarning[] {
    const warnings: LanguageModelV2CallWarning[] = [];

    const unsupportedSettings = [
      { value: topK, name: "topK" },
      { value: seed, name: "seed" },
      { value: presencePenalty, name: "presencePenalty" },
      { value: frequencyPenalty, name: "frequencyPenalty" },
      { value: stopSequences, name: "stopSequences" },
    ] as const;

    for (const { value, name } of unsupportedSettings) {
      if (value != null) {
        warnings.push({ type: "unsupported-setting", setting: name });
      }
    }

    return warnings;
  }

  private async parseProviderOptions(providerOptions: any): Promise<OllamaResponsesProviderOptions | null> {
    const result = await parseProviderOptions({
      provider: "ollama",
      providerOptions,
      schema: ollamaProviderOptions,
    });
    return result ?? null;
  }

  private buildBaseArgs({
    modelId,
    prompt,
    temperature,
    topP,
    maxOutputTokens,
    responseFormat,
    ollamaOptions,
  }: {
    modelId: OllamaChatModelId;
    prompt: any;
    temperature?: number;
    topP?: number;
    maxOutputTokens?: number;
    responseFormat?: any;
    ollamaOptions: OllamaResponsesProviderOptions | null;
  }) {
    return {
      model: modelId,
      messages: convertToOllamaChatMessages({
        prompt,
        systemMessageMode: "system",
      }),
      temperature,
      top_p: topP,
      max_output_tokens: maxOutputTokens,

      ...(responseFormat?.type === "json" && {
        format: responseFormat.schema != null ? responseFormat.schema : "json",
      }),

      think: ollamaOptions?.think ?? false,
      options: ollamaOptions?.options?? undefined
    };
  }
} 