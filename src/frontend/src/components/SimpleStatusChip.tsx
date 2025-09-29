interface StatusChipProps {
  label: string;
  value: string | number;
  variant?: 'success' | 'live' | 'default';
  icon?: string;
}

export default function SimpleStatusChip({ label, value, variant = 'default', icon }: StatusChipProps) {
  const getVariantClasses = () => {
    switch (variant) {
      case 'success':
        return 'bg-success/10 text-success border-success/20';
      case 'live':
        return 'bg-primary/10 text-primary border-primary/20';
      default:
        return 'bg-muted text-foreground border-border';
    }
  };

  return (
    <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-lg border text-sm font-medium ${getVariantClasses()}`}>
      {icon && <span className="material-icon text-sm">{icon}</span>}
      <div>
        <div className="font-semibold">{value}</div>
        <div className="text-xs opacity-80">{label}</div>
      </div>
    </div>
  );
}