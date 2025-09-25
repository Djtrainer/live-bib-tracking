interface IconButtonProps {
  icon: string;
  onClick?: () => void;
  title?: string;
  variant?: 'default' | 'destructive' | 'success';
  disabled?: boolean;
}

export default function IconButton({ 
  icon, 
  onClick, 
  title, 
  variant = 'default',
  disabled = false 
}: IconButtonProps) {
  const variantClasses = {
    default: 'hover:text-accent',
    destructive: 'hover:text-destructive',
    success: 'hover:text-success'
  };

  const variantStyles = {
    default: '',
    destructive: 'hover:text-destructive hover:bg-destructive/10 hover:border-destructive/20',
    success: 'hover:text-success hover:bg-success/10 hover:border-success/20'
  };

  return (
    <button
      className={`icon-button ${variantStyles[variant]} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      onClick={onClick}
      title={title}
      disabled={disabled}
    >
      <span className="material-icon">{icon}</span>
    </button>
  );
}