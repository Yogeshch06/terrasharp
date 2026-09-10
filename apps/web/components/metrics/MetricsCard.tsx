interface MetricsCardProps {
  label: string;
  value: string | number;
  unit?: string;
  subtitle?: string;
  highlight?: boolean;
}

export default function MetricsCard({ label, value, unit, subtitle, highlight = false }: MetricsCardProps) {
  const numVal = typeof value === "number" ? value : parseFloat(String(value));
  const displayVal = isNaN(numVal) ? String(value) : numVal.toFixed(numVal < 10 ? 4 : 2);

  return (
    <div className={`border p-5 space-y-1 ${highlight ? "border-primary bg-primary/5" : "border-border bg-card"}`}>
      <div className="text-xs text-muted-foreground uppercase tracking-wide">{label}</div>
      <div className="text-3xl font-bold text-foreground tabular-nums">
        {displayVal}
        {unit && <span className="text-lg font-normal text-muted-foreground ml-1">{unit}</span>}
      </div>
      {subtitle && <div className="text-xs text-muted-foreground">{subtitle}</div>}
    </div>
  );
}
