import {
  EmbeddingModelV1,
  ImageModelV1,
  LanguageModelV1,
  ProviderV1,
} from '@ai-sdk/provider';
import {
  FetchFunction,
  loadApiKey,
  withoutTrailingSlash,
} from '@ai-sdk/provider-utils';
import { OllamaChatLanguageModel } from './ollama-chat-language-model';
import { OllamaChatModelId, OllamaChatSettings } from './ollama-chat-settings';
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

export interface OllamaProvider extends ProviderV1 {
  (
    modelId: 'gpt-3.5-turbo-instruct',
    settings?: OllamaCompletionSettings,
  ): OllamaCompletionLanguageModel;
  (modelId: OllamaChatModelId, settings?: OllamaChatSettings): LanguageModelV1;

  /**
Creates an Ollama model for text generation.
   */
  languageModel(
    modelId: 'gpt-3.5-turbo-instruct',
    settings?: OllamaCompletionSettings,
  ): OllamaCompletionLanguageModel;
  languageModel(
    modelId: OllamaChatModelId,
    settings?: OllamaChatSettings,
  ): LanguageModelV1;

  /**
Creates an Ollama chat model for text generation.
   */
  chat(
    modelId: OllamaChatModelId,
    settings?: OllamaChatSettings,
  ): LanguageModelV1;

  /**
Creates an Ollama completion model for text generation.
   */
  completion(
    modelId: OllamaCompletionModelId,
    settings?: OllamaCompletionSettings,
  ): LanguageModelV1;

  /**
Creates a model for text embeddings.
   */
  embedding(
    modelId: OllamaEmbeddingModelId,
    settings?: OllamaEmbeddingSettings,
  ): EmbeddingModelV1<string>;

  /**
Creates a model for text embeddings.

@deprecated Use `textEmbeddingModel` instead.
   */
  textEmbedding(
    modelId: OllamaEmbeddingModelId,
    settings?: OllamaEmbeddingSettings,
  ): EmbeddingModelV1<string>;

  /**
Creates a model for text embeddings.
   */
  textEmbeddingModel(
    modelId: OllamaEmbeddingModelId,
    settings?: OllamaEmbeddingSettings,
  ): EmbeddingModelV1<string>;

  /**
Creates a model for image generation.
   */
  image(
    modelId: OllamaImageModelId,
    settings?: OllamaImageSettings,
  ): ImageModelV1;

  /**
Creates a model for image generation.
   */
  imageModel(
    modelId: OllamaImageModelId,
    settings?: OllamaImageSettings,
  ): ImageModelV1;
}

export interface OllamaProviderSettings {
  /**
Base URL for the Ollama API calls.
     */
  baseURL?: string;

  /**
API key for authenticating requests.
     */
  apiKey?: string;

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
    withoutTrailingSlash(options.baseURL) ?? 'https://api.ollama.com/v1';

  // we default to compatible, because strict breaks providers like Groq:
  const compatibility = options.compatibility ?? 'compatible';

  const providerName = options.name ?? 'ollama';

  const getHeaders = () => ({
    Authorization: `Bearer ${loadApiKey({
      apiKey: options.apiKey,
      environmentVariableName: 'OPENAI_API_KEY',
      description: 'Ollama',
    })}`,
    'Ollama-Organization': options.organization,
    'Ollama-Project': options.project,
    ...options.headers,
  });

  const createChatModel = (
    modelId: OllamaChatModelId,
    settings: OllamaChatSettings = {},
  ) =>
    new OllamaChatLanguageModel(modelId, settings, {
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
    modelId: OllamaChatModelId | OllamaCompletionModelId,
    settings?: OllamaChatSettings | OllamaCompletionSettings,
  ) => {
    if (new.target) {
      throw new Error(
        'The Ollama model function cannot be called with the new keyword.',
      );
    }

    if (modelId === 'gpt-3.5-turbo-instruct') {
      return createCompletionModel(
        modelId,
        settings as OllamaCompletionSettings,
      );
    }

    return createChatModel(modelId, settings as OllamaChatSettings);
  };

  const provider = function (
    modelId: OllamaChatModelId | OllamaCompletionModelId,
    settings?: OllamaChatSettings | OllamaCompletionSettings,
  ) {
    return createLanguageModel(modelId, settings);
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
