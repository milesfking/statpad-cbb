export function displayConference(name: string | null | undefined): string {
  if (!name) return "—";
  const trimmed = name.trim();
  if (/ conference$/i.test(trimmed)) {
    return trimmed.replace(/\s+conference$/i, "").trim();
  }
  return trimmed;
}

/** US-style M/D/YYYY (e.g. 11/11/2026). */
export function formatShortDate(input: string | Date): string {
  const d = typeof input === "string" ? new Date(input) : input;
  if (Number.isNaN(d.getTime())) return "—";
  return `${d.getMonth() + 1}/${d.getDate()}/${d.getFullYear()}`;
}

export function formatAsOfLine(asOf: string | undefined): string {
  if (!asOf) return "";
  const d = new Date(asOf.includes("T") ? asOf : `${asOf}T12:00:00`);
  return `Rankings updated through ${formatShortDate(d)}`;
}

export function formatSiteLastUpdated(generatedAt: string): string {
  return `Site last updated ${formatShortDate(generatedAt)}`;
}
