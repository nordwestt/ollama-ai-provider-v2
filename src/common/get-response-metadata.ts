export function getResponseMetadata({
  model,
  created_at,
}: {
  created_at?: string | undefined | null;
  model?: string | undefined | null;
}) {
  return {
    id: undefined,
    modelId: model ?? undefined,
    timestamp: created_at != null ? new Date(created_at) : undefined,
  };
}
