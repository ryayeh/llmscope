import type { ComponentProps, ReactNode } from "react";

import { cn } from "@/lib/utils";

interface PanelProps extends ComponentProps<"section"> {
  title: string;
  description?: string;
  action?: ReactNode;
}

export function Panel({
  action,
  children,
  className,
  description,
  title,
  ...props
}: PanelProps) {
  return (
    <section
      className={cn(
        "rounded-3xl border border-slate-200 bg-white/88 p-5 shadow-[0_24px_60px_-40px_rgba(15,23,42,0.35)] backdrop-blur sm:p-6",
        className,
      )}
      {...props}
    >
      <div className="flex flex-col gap-4 border-b border-slate-200 pb-5 sm:flex-row sm:items-start sm:justify-between">
        <div className="max-w-2xl">
          <h3 className="text-lg font-semibold text-slate-950">{title}</h3>
          {description ? (
            <p className="mt-1 text-sm leading-6 text-slate-500">{description}</p>
          ) : null}
        </div>
        {action ? <div className="shrink-0">{action}</div> : null}
      </div>

      <div className="pt-5">{children}</div>
    </section>
  );
}
