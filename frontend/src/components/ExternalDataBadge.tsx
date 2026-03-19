import { AlertTriangle, CheckCircle2 } from "lucide-react";

interface ExternalDataBadgeProps {
  enabled: boolean;
  size?: "sm" | "md";
}

export function ExternalDataBadge({ enabled, size = "md" }: ExternalDataBadgeProps) {
  if (enabled) {
    return (
      <span className="inline-flex items-center gap-1.5 rounded-full bg-emerald-100 px-3 py-1 text-xs font-semibold text-emerald-700 ring-1 ring-emerald-200">
        <CheckCircle2 className={size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5"} />
        External data ON
      </span>
    );
  }

  return (
    <span className="inline-flex items-center gap-1.5 rounded-full bg-orange-100 px-3 py-1 text-xs font-semibold text-orange-700 ring-1 ring-orange-200">
      <AlertTriangle className={size === "sm" ? "h-3 w-3" : "h-3.5 w-3.5"} />
      External data OFF
    </span>
  );
}

/** Full-width warning banner shown at the top of the detail page */
export function ExternalDataWarning() {
  return (
    <div className="flex items-start gap-3 rounded-lg border border-orange-200 bg-orange-50 px-4 py-3 text-sm text-orange-800">
      <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-orange-500" />
      <div>
        <span className="font-semibold">External data is disabled.</span>
        {" "}Only technical indicators were used as features. Enable{" "}
        <code className="rounded bg-orange-100 px-1 py-0.5 text-xs font-mono">
          external_data.enabled: true
        </code>{" "}
        in <code className="rounded bg-orange-100 px-1 py-0.5 text-xs font-mono">config/default.yaml</code>{" "}
        to add macro / market features.
      </div>
    </div>
  );
}
