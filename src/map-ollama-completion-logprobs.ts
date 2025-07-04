import { LanguageModelV2LogProbs } from '@ai-sdk/provider';

type OllamaCompletionLogProps = {
  tokens: string[];
  token_logprobs: number[];
  top_logprobs: Record<string, number>[] | null;
};

export function mapOllamaCompletionLogProbs(
  logprobs: OllamaCompletionLogProps | null | undefined,
): LanguageModelV2LogProbs | undefined {
  return logprobs?.tokens.map((token, index) => ({
    token,
    logprob: logprobs.token_logprobs[index],
    topLogprobs: logprobs.top_logprobs
      ? Object.entries(logprobs.top_logprobs[index]).map(
          ([token, logprob]) => ({
            token,
            logprob,
          }),
        )
      : [],
  }));
}
