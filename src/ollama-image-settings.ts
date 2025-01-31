export type OllamaImageModelId = 'dall-e-3' | 'dall-e-2' | (string & {});

// https://platform.ollama.com/docs/guides/images
export const modelMaxImagesPerCall: Record<OllamaImageModelId, number> = {
  'dall-e-3': 1,
  'dall-e-2': 10,
};

export interface OllamaImageSettings {
  /**
Override the maximum number of images per call (default is dependent on the
model, or 1 for an unknown model).
   */
  maxImagesPerCall?: number;
}
