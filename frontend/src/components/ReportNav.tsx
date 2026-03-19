import Link from "next/link";
import { BarChart2 } from "lucide-react";

interface ReportNavProps {
  activePage?: "reports" | "compare";
}

export function ReportNav({ activePage }: ReportNavProps) {
  return (
    <nav className="border-b bg-white">
      <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-4">
        <Link href="/reports" className="flex items-center gap-2">
          <BarChart2 className="h-5 w-5 text-slate-700" />
          <span className="text-sm font-semibold text-slate-900">
            Samsung Forecast
          </span>
        </Link>
        <div className="flex items-center gap-6 text-sm font-medium">
          <Link
            href="/reports"
            className={
              activePage === "reports"
                ? "text-slate-900 border-b-2 border-slate-900 pb-0.5"
                : "text-slate-500 hover:text-slate-900 transition-colors"
            }
          >
            Reports
          </Link>
          <Link
            href="/compare"
            className={
              activePage === "compare"
                ? "text-slate-900 border-b-2 border-slate-900 pb-0.5"
                : "text-slate-500 hover:text-slate-900 transition-colors"
            }
          >
            Compare
          </Link>
        </div>
      </div>
    </nav>
  );
}
