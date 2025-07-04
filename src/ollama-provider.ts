import {
  EmbeddingModelV2,
  ImageModelV2,
  LanguageModelV2,
  ProviderV2,
  TranscriptionModelV2,
  SpeechModelV2,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  withoutTrailingSlash,
} from '@ai-sdk/provider-utils';
import { OllamaChatLanguageModel } from './ollama-chat-language-model';
import { OllamaChatModelId, ollamaProviderOptions } from './ollama-chat-settings';
import { OllamaCompletionLanguageModel } from './ollama-completion-language-model';
import {
  OllamaCompletionModelId,
  OllamaCompletionSettings,
} from './ollama-completion-settings';
import { OllamaEmbeddingModel } from './ollama-embedding-model';
import {
  OllamaEmbeddingModelId,
  OllamaEmbeddingSettings,
} from './ollama-embedding-settings';
import { OllamaImageModel } from './ollama-image-model';
import {
  OllamaImageModelId,
  OllamaImageSettings,
} from './ollama-image-settings';
import { OllamaResponsesModelId } from './ollama-responses-settings';
import { OllamaResponsesLanguageModel } from './responses/ollama-responses-language-model';

export interface OllamaProvider extends ProviderV2 {
  (modelId: OllamaChatModelId): LanguageModelV2;

  /**
Creates an Ollama model for text generation.
   */
  languageModel(modelId: OllamaResponsesModelId): OllamaResponsesLanguageModel;

  /**
Creates an Ollama chat model for text generation.
   */
  chat(
    modelId: OllamaChatModelId,
    settings?: typeof ollamaProviderOptions,
  ): LanguageModelV2;

  /**
Creates an Ollama completion model for text generation.
   */
  completion(
    modelId: OllamaCompletionModelId,
    settings?: OllamaCompletionSettings,
  ): LanguageModelV2;

  /**
Creates a model for text embeddings.
   */
  embedding(
    modelId: OllamaEmbeddingModelId,
    settings?: OllamaEmbeddingSettings,
  ): EmbeddingModelV2<string>;

  /**
Creates a model for text embeddings.

@deprecated Use `textEmbeddingModel` instead.
   */
  textEmbedding(
    modelId: OllamaEmbeddingModelId,
    settings?: OllamaEmbeddingSettings,
  ): EmbeddingModelV2<string>;

  /**
Creates a model for text embeddings.
   */
  textEmbeddingModel(
    modelId: OllamaEmbeddingModelId,
    settings?: OllamaEmbeddingSettings,
  ): EmbeddingModelV2<string>;

  /**
Creates a model for image generation.
   */
  image(
    modelId: OllamaImageModelId,
    settings?: OllamaImageSettings,
  ): ImageModelV2;

  /**
Creates a model for image generation.
   */
  imageModel(
    modelId: OllamaImageModelId,
    settings?: OllamaImageSettings,
  ): ImageModelV2;
}

export interface OllamaProviderSettings {
  /**
Base URL for the Ollama API calls.
     */
  baseURL?: string;

  /**
Ollama Organization.
     */
  organization?: string;

  /**
Ollama project.
     */
  project?: string;

  /**
Custom headers to include in the requests.
     */
  headers?: Record<string, string>;

  /**
Ollama compatibility mode. Should be set to `strict` when using the Ollama API,
and `compatible` when using 3rd party providers. In `compatible` mode, newer
information such as streamOptions are not being sent. Defaults to 'compatible'.
   */
  compatibility?: 'strict' | 'compatible';

  /**
Provider name. Overrides the `ollama` default name for 3rd party providers.
   */
  name?: string;

  /**
Custom fetch implementation. You can use it as a middleware to intercept requests,
or to provide a custom fetch implementation for e.g. testing.
    */
  fetch?: FetchFunction;
}

/**
Create an Ollama provider instance.
 */
export function createOllama(
  options: OllamaProviderSettings = {},
): OllamaProvider {
  const baseURL =
    withoutTrailingSlash(options.baseURL) ?? 'http://127.0.0.1:11434/api';

  // we default to compatible, because strict breaks providers like Groq:
  const compatibility = options.compatibility ?? 'compatible';

  const providerName = options.name ?? 'ollama';

  const getHeaders = () => ({
    'Ollama-Organization': options.organization,
    'Ollama-Project': options.project,
    ...options.headers,
  });

  const createChatModel = (modelId: OllamaChatModelId) =>
    new OllamaChatLanguageModel(modelId, {
      provider: `${providerName}.chat`,
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      compatibility,
      fetch: options.fetch,
    });

  const createCompletionModel = (
    modelId: OllamaCompletionModelId,
    settings: OllamaCompletionSettings = {},
  ) =>
    new OllamaCompletionLanguageModel(modelId, settings, {
      provider: `${providerName}.completion`,
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      compatibility,
      fetch: options.fetch,
    });

  const createEmbeddingModel = (
    modelId: OllamaEmbeddingModelId,
    settings: OllamaEmbeddingSettings = {},
  ) =>
    new OllamaEmbeddingModel(modelId, settings, {
      provider: `${providerName}.embedding`,
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createImageModel = (
    modelId: OllamaImageModelId,
    settings: OllamaImageSettings = {},
  ) =>
    new OllamaImageModel(modelId, settings, {
      provider: `${providerName}.image`,
      url: ({ path }) => `${baseURL}${path}`,
      headers: getHeaders,
      fetch: options.fetch,
    });

  const createLanguageModel = (
    modelId: OllamaChatModelId | OllamaCompletionModelId) => {
    if (new.target) {
      throw new Error(
        'The Ollama model function cannot be called with the new keyword.',
      );
    }

    return createChatModel(modelId);
  };

  const provider = function (modelId: OllamaChatModelId | OllamaCompletionModelId) {
    return createLanguageModel(modelId);
  };

  provider.languageModel = createLanguageModel;
  provider.chat = createChatModel;
  provider.completion = createCompletionModel;
  provider.embedding = createEmbeddingModel;
  provider.textEmbedding = createEmbeddingModel;
  provider.textEmbeddingModel = createEmbeddingModel;

  provider.image = createImageModel;
  provider.imageModel = createImageModel;

  return provider as OllamaProvider;
}

/**
Default Ollama provider instance. It uses 'strict' compatibility mode.
 */
export const ollama = createOllama({
  compatibility: 'strict', // strict for Ollama API
});
