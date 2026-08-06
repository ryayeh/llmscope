"use client";

interface TokenContextAction {
  destructive?: boolean;
  disabled?: boolean;
  label: string;
  onSelect: () => void;
}

interface TokenContextMenuProps {
  actions: TokenContextAction[];
  onClose: () => void;
  title: string;
  x: number;
  y: number;
}

export function TokenContextMenu({
  actions,
  onClose,
  title,
  x,
  y,
}: TokenContextMenuProps) {
  return (
    <div
      className="context-menu"
      style={{
        left: x,
        top: y,
      }}
    >
      <div className="context-menu__title">{title}</div>
      {actions.map((action) => (
        <button
          key={action.label}
          className={`context-menu__action${action.destructive ? " context-menu__action--destructive" : ""}`}
          disabled={action.disabled}
          onClick={action.onSelect}
          type="button"
        >
          {action.label}
        </button>
      ))}
      <button className="context-menu__action context-menu__action--muted" onClick={onClose} type="button">
        Close
      </button>
    </div>
  );
}
