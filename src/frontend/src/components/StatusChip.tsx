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
      {icon && <span className="material-icon text-base">{icon}</span>}
      <div className="flex flex-col items-start">
        <span className="text-xs opacity-80 font-medium">{label}</span>
        <span className="font-bold text-sm">{value}</span>
      </div>
    </div>
  );
}