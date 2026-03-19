import { cn } from "@/lib/utils";
import { verdictStyle } from "@/lib/report-utils";

interface VerdictBadgeProps {
  verdict: string;
  size?: "sm" | "md" | "lg";
}

export function VerdictBadge({ verdict, size = "md" }: VerdictBadgeProps) {
  const s = verdictStyle(verdict);
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full font-semibold ring-1",
        s.bg,
        s.text,
        s.ring,
        size === "sm" && "px-2 py-0.5 text-xs",
        size === "md" && "px-3 py-1 text-sm",
        size === "lg" && "px-4 py-1.5 text-base"
      )}
    >
      {s.label}
    </span>
  );
}
