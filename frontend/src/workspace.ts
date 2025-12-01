export function getWorkspaceId(): string {
  const KEY = "ai_ks_workspace_id";
  let current = localStorage.getItem(KEY);
  if (!current) {
    current = `ws_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
    localStorage.setItem(KEY, current);
  }
  return current;
}
