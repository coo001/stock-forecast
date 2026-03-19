import { cn } from "@/lib/utils";
import { Card, CardContent } from "@/components/ui/card";

interface MetricCardProps {
  title: string;
  value: string;
  subtitle?: string;
  valueClass?: string;
}

export function MetricCard({ title, value, subtitle, valueClass }: MetricCardProps) {
  return (
    <Card>
      <CardContent className="pt-6">
        <p className="text-xs font-medium uppercase tracking-wider text-slate-500">
          {title}
        </p>
        <p className={cn("mt-1 text-3xl font-bold tabular-nums", valueClass ?? "text-slate-900")}>
          {value}
        </p>
        {subtitle && (
          <p className="mt-1 text-xs text-slate-400">{subtitle}</p>
        )}
      </CardContent>
    </Card>
  );
}
