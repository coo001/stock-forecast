/**
 * Server-side data access layer.
 * Reads experiment report JSON files from the reports/ directory.
 *
 * Default path: ../reports/ relative to the Next.js project root (frontend/).
 * Override with REPORTS_DIR environment variable.
 */
import { readdir, readFile } from "fs/promises";
import path from "path";
import type { ExperimentReport, ReportListItem } from "./types";
import { toListItem } from "./report-utils";

function reportsDir(): string {
  return (
    process.env.REPORTS_DIR ?? path.join(process.cwd(), "..", "reports")
  );
}

export async function listReports(): Promise<ReportListItem[]> {
  const dir = reportsDir();
  let files: string[];
  try {
    files = await readdir(dir);
  } catch {
    return [];
  }

  const jsonFiles = files.filter(
    (f) => f.endsWith(".json") && f.startsWith("exp_")
  );

  const items = await Promise.allSettled(
    jsonFiles.map(async (file) => {
      const raw = await readFile(path.join(dir, file), "utf-8");
      return toListItem(JSON.parse(raw) as ExperimentReport);
    })
  );

  return items
    .filter((r): r is PromiseFulfilledResult<ReportListItem> => r.status === "fulfilled")
    .map((r) => r.value)
    .sort((a, b) => b.generated_at.localeCompare(a.generated_at));
}

export async function getReport(id: string): Promise<ExperimentReport | null> {
  // Sanitise: only allow alphanumeric, underscores, hyphens
  if (!/^[\w-]+$/.test(id)) return null;
  try {
    const filePath = path.join(reportsDir(), `${id}.json`);
    const raw = await readFile(filePath, "utf-8");
    return JSON.parse(raw) as ExperimentReport;
  } catch {
    return null;
  }
}
