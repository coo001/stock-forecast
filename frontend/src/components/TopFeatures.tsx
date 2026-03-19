import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { TrendingUp } from "lucide-react";

interface TopFeaturesProps {
  topFeatures: string[];
  allFeatures: string[];
}

const FEATURE_LABELS: Record<string, string> = {
  log_ret_1d: "1d Return",
  log_ret_5d: "5d Return",
  log_ret_10d: "10d Return",
  log_ret_15d: "15d Return",
  log_ret_20d: "20d Return",
  ma_ratio_5d: "MA Ratio 5d",
  ma_ratio_10d: "MA Ratio 10d",
  ma_ratio_20d: "MA Ratio 20d",
  ma_ratio_30d: "MA Ratio 30d",
  ma_ratio_50d: "MA Ratio 50d",
  ma_ratio_60d: "MA Ratio 60d",
  rsi_14: "RSI (14)",
  atr_pct_14: "ATR% (14)",
  vol_ratio_20d: "Vol Ratio 20d",
  vol_ratio_30d: "Vol Ratio 30d",
};

function featureLabel(name: string): string {
  if (FEATURE_LABELS[name]) return FEATURE_LABELS[name];
  if (name.startsWith("ext_")) return name.replace("ext_", "").toUpperCase();
  return name;
}

export function TopFeatures({ topFeatures, allFeatures }: TopFeaturesProps) {
  const isExternal = (name: string) => name.startsWith("ext_");

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-base">
          <TrendingUp className="h-4 w-4 text-slate-500" />
          Feature Importance
        </CardTitle>
        <p className="text-xs text-slate-400">
          {allFeatures.length} features total · top 5 by LightGBM split gain
        </p>
      </CardHeader>
      <CardContent>
        <ol className="space-y-2">
          {topFeatures.map((name, i) => (
            <li key={name} className="flex items-center gap-3">
              <span className="flex h-6 w-6 shrink-0 items-center justify-center rounded-full bg-slate-100 text-xs font-bold text-slate-500">
                {i + 1}
              </span>
              <div className="flex flex-1 items-center justify-between rounded-md bg-slate-50 px-3 py-1.5">
                <span className="text-sm font-medium text-slate-800">
                  {featureLabel(name)}
                </span>
                <code className="text-xs text-slate-400">{name}</code>
              </div>
              {isExternal(name) && (
                <span className="rounded-full bg-blue-100 px-2 py-0.5 text-xs font-medium text-blue-700">
                  ext
                </span>
              )}
            </li>
          ))}
        </ol>

        {allFeatures.length > 0 && (
          <div className="mt-4 border-t pt-4">
            <p className="mb-2 text-xs font-medium text-slate-500 uppercase tracking-wider">
              All features
            </p>
            <div className="flex flex-wrap gap-1.5">
              {allFeatures.map((name) => (
                <span
                  key={name}
                  className={`rounded-full px-2 py-0.5 text-xs font-mono ${
                    isExternal(name)
                      ? "bg-blue-100 text-blue-700"
                      : topFeatures.includes(name)
                      ? "bg-slate-800 text-slate-100"
                      : "bg-slate-100 text-slate-600"
                  }`}
                >
                  {name}
                </span>
              ))}
            </div>
          </div>
        )}
      </CardContent>
    </Card>
  );
}
