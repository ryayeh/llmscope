"use client";

interface TokenContextMenuProps {
  canCollapse: boolean;
  canExpand: boolean;
  onCenter: () => void;
  onClose: () => void;
  onCollapse: () => void;
  onExpand: () => void;
  title: string;
  x: number;
  y: number;
}

export function TokenContextMenu({
  canCollapse,
  canExpand,
  onCenter,
  onClose,
  onCollapse,
  onExpand,
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
      {canExpand ? (
        <button className="context-menu__action" onClick={onExpand} type="button">
          Expand futures
        </button>
      ) : null}
      {canCollapse ? (
        <button className="context-menu__action" onClick={onCollapse} type="button">
          Collapse subtree
        </button>
      ) : null}
      <button className="context-menu__action" onClick={onCenter} type="button">
        Center
      </button>
      <button className="context-menu__action context-menu__action--muted" onClick={onClose} type="button">
        Close
      </button>
    </div>
  );
}
