"use client";

import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import type { FoldSummary } from "@/lib/types";
import { toChartData } from "@/lib/report-utils";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface SingleChartProps {
  data: ReturnType<typeof toChartData>;
  dataKey: "da" | "sharpe" | "rmse" | "ic";
  title: string;
  color: string;
  refLine?: number;
  refLineLabel?: string;
  unit?: string;
  domain?: [number | "auto", number | "auto"];
}

function SingleLineChart({
  data,
  dataKey,
  title,
  color,
  refLine,
  refLineLabel,
  unit = "",
  domain = ["auto", "auto"],
}: SingleChartProps) {
  return (
    <Card>
      <CardHeader className="pb-2">
        <CardTitle className="text-sm font-medium text-slate-600">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent className="pb-4">
        <ResponsiveContainer width="100%" height={200}>
          <LineChart
            data={data}
            margin={{ top: 4, right: 8, left: -16, bottom: 0 }}
          >
            <CartesianGrid strokeDasharray="3 3" stroke="#f1f5f9" />
            <XAxis
              dataKey="label"
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={domain}
              tick={{ fontSize: 10, fill: "#94a3b8" }}
              tickLine={false}
              axisLine={false}
              tickFormatter={(v: number) =>
                isFinite(v) ? `${v.toFixed(1)}${unit}` : "—"
              }
            />
            <Tooltip
              contentStyle={{
                fontSize: 12,
                borderRadius: 8,
                border: "1px solid #e2e8f0",
                boxShadow: "0 2px 8px rgba(0,0,0,0.08)",
              }}
              formatter={(v: number) => [
                isFinite(v) ? `${v.toFixed(3)}${unit}` : "—",
                title,
              ]}
              labelFormatter={(l) => `Fold ${l}`}
            />
            {refLine !== undefined && (
              <ReferenceLine
                y={refLine}
                stroke="#cbd5e1"
                strokeDasharray="4 3"
                label={
                  refLineLabel
                    ? { value: refLineLabel, fontSize: 10, fill: "#94a3b8", position: "insideTopRight" }
                    : undefined
                }
              />
            )}
            <Line
              type="monotone"
              dataKey={dataKey}
              stroke={color}
              strokeWidth={2}
              dot={{ r: 2, fill: color, strokeWidth: 0 }}
              activeDot={{ r: 4, strokeWidth: 0 }}
              connectNulls={false}
            />
          </LineChart>
        </ResponsiveContainer>
      </CardContent>
    </Card>
  );
}

interface FoldChartsProps {
  folds: FoldSummary[];
}

export function FoldCharts({ folds }: FoldChartsProps) {
  const data = toChartData(folds);

  return (
    <div className="grid grid-cols-1 gap-4 md:grid-cols-2 xl:grid-cols-4">
      <SingleLineChart
        data={data}
        dataKey="da"
        title="Directional Accuracy (%)"
        color="#3b82f6"
        refLine={50}
        refLineLabel="50%"
        unit="%"
        domain={[30, 70]}
      />
      <SingleLineChart
        data={data}
        dataKey="sharpe"
        title="Sharpe Ratio"
        color="#8b5cf6"
        refLine={0}
        unit=""
      />
      <SingleLineChart
        data={data}
        dataKey="rmse"
        title="RMSE (× 100)"
        color="#f59e0b"
        unit=""
      />
      <SingleLineChart
        data={data}
        dataKey="ic"
        title="Information Coefficient"
        color="#10b981"
        refLine={0}
        unit=""
      />
    </div>
  );
}
