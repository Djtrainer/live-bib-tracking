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
    success: "text-success",
    warning: "text-warning"
  };

  return (
    <div className={`${baseClasses} ${variantClasses[variant]}`}>
      {icon && <span className="material-icon text-xs">{icon}</span>}
      <span className="font-medium">{label}:</span>
      <span>{value}</span>
    </div>
  );
}