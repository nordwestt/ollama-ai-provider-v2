import { JSONSchema7 } from '@ai-sdk/provider';

export type OllamaTools = Array<{
    type: 'function';
    function: {
      name: string;
      description: string | undefined;
      parameters: JSONSchema7;
      strict: boolean | undefined;
    };
  }>;

  export type OllamaToolChoice =
  | 'auto'
  | 'none'
  | 'required'
  | { type: 'function'; function: { name: string } };