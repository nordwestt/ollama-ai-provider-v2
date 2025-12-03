import { JSONSchema7 } from '@ai-sdk/provider';

export type OllamaResponsesPrompt = Array<OllamaResponsesMessage>;

export type OllamaResponsesMessage =
  | OllamaResponsesSystemMessage
  | OllamaResponsesUserMessage
  | OllamaResponsesAssistantMessage
  | OllamaResponsesFunctionCall
  | OllamaResponsesFunctionCallOutput
  | OllamaWebSearchCall
  | OllamaComputerCall;

export type OllamaResponsesSystemMessage = {
  role: 'system' | 'developer';
  content: string;
};

export type OllamaResponsesUserMessage = {
  role: 'user';
  content: Array<
    | { type: 'input_text'; text: string }
    | { type: 'input_image'; image_url: string }
    | { type: 'input_file'; filename: string; file_data: string }
  >;
};

export type OllamaResponsesAssistantMessage = {
  role: 'assistant';
  content: Array<
    | { type: 'output_text'; text: string }
    | OllamaWebSearchCall
    | OllamaComputerCall
  >;
};

export type OllamaResponsesFunctionCall = {
  type: 'function_call';
  call_id: string;
  name: string;
  arguments: string;
};

export type OllamaResponsesFunctionCallOutput = {
  type: 'function_call_output';
  call_id: string;
  output: string;
};

export type OllamaWebSearchCall = {
  type: 'web_search_call';
  id: string;
  status?: string;
};

export type OllamaComputerCall = {
  type: 'computer_call';
  id: string;
  status?: string;
};

export type OllamaResponsesTool =
  | {
      type: 'function';
      function: {
        name: string;
        description: string | undefined;
        parameters: JSONSchema7;
      };
    }
  | {
      type: 'web_search_preview';
      search_context_size: 'low' | 'medium' | 'high';
      user_location: {
        type: 'approximate';
        city: string;
        region: string;
      };
    };
