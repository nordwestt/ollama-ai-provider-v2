import { LanguageModelV2LogProbs } from '@ai-sdk/provider';

type OllamaChatLogProbs = {
  content:
    | {
        token: string;
        logprob: number;
        top_logprobs:
          | {
              token: string;
              logprob: number;
            }[]
          | null;
      }[]
    | null;
};

export function mapOllamaChatLogProbsOutput(
  logprobs: OllamaChatLogProbs | null | undefined,
): LanguageModelV2LogProbs | undefined {
  return (
    logprobs?.content?.map(({ token, logprob, top_logprobs }) => ({
      token,
      logprob,
      topLogprobs: top_logprobs
        ? top_logprobs.map(({ token, logprob }) => ({
            token,
            logprob,
          }))
        : [],
    })) ?? undefined
  );
}
