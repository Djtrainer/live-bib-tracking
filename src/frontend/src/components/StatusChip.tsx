interface StatusChipProps {
  label: string;
  value: string | number;
  variant?: 'default' | 'live' | 'success' | 'warning';
  icon?: string;
}

export default function StatusChip({ label, value, variant = 'default', icon }: StatusChipProps) {
  const baseClasses = "status-chip";
  const variantClasses = {
    default: "",
    live: "live",
    success: "success",
    warning: "warning"
  };

  return (
    <div className={`${baseClasses} ${variantClasses[variant]}`}>
      {icon && <span className="material-icon text-lg">{icon}</span>}
      <div className="flex flex-col">
        <span className="text-xs font-medium opacity-80 uppercase tracking-wider">{label}</span>
        <span className="font-bold text-lg leading-tight">{value}</span>
      </div>
    </div>
  );
}