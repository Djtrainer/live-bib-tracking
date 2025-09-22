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

  return (
    <button
      className={`icon-button ${variantClasses[variant]} ${disabled ? 'opacity-50 cursor-not-allowed' : ''}`}
      onClick={onClick}
      title={title}
      disabled={disabled}
    >
      <span className="material-icon text-base">{icon}</span>
    </button>
  );
}