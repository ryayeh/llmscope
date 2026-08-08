"use client";

import { useEffect, useRef, useState } from "react";
import { ChevronDown, ChevronRight, Move } from "lucide-react";

import { cn } from "@/lib/utils";
import type { TokenTreeNode } from "@/types/api";

interface TokenBranchTreeProps {
  root: TokenTreeNode;
  selectedNodeId: string | null;
  expandedNodeIds: string[];
  searchQuery: string;
  onSelectNode: (nodeId: string) => void;
  onToggleNode: (nodeId: string) => void;
}

interface DragState {
  dragging: boolean;
  pointerId: number | null;
  startLeft: number;
  startTop: number;
  startX: number;
  startY: number;
}

interface HeadAccent {
  label: string;
  rgb: string;
  textClass: string;
}

interface HeadSummary {
  headIndex: number;
  node: TokenTreeNode;
  visibleCount: number;
}

interface GridNode {
  childCount: number;
  hasVisibleChildren: boolean;
  headIndex: number;
  matchesSearch: boolean;
  node: TokenTreeNode;
  parentLabel: string;
}

interface GridRow {
  cells: GridNode[][];
  depth: number;
  label: string;
}

const HEAD_ACCENTS: HeadAccent[] = [
  { label: "Head 1", rgb: "14 165 233", textClass: "text-sky-700" },
  { label: "Head 2", rgb: "16 185 129", textClass: "text-emerald-700" },
  { label: "Head 3", rgb: "245 158 11", textClass: "text-amber-700" },
  { label: "Head 4", rgb: "168 85 247", textClass: "text-violet-700" },
];

function formatPercent(value: number | null | undefined) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    return "Unavailable";
  }

  return `${(value * 100).toFixed(value < 0.1 ? 1 : 0)}%`;
}

function nodeMatchesSearch(node: TokenTreeNode, searchQuery: string) {
  const query = searchQuery.trim().toLowerCase();

  if (!query) {
    return true;
  }

  return (
    node.token.toLowerCase().includes(query) ||
    node.text_preview.toLowerCase().includes(query)
  );
}

function nodeHasVisibleDescendant(node: TokenTreeNode, searchQuery: string): boolean {
  if (nodeMatchesSearch(node, searchQuery)) {
    return true;
  }

  return node.children.some((child) => nodeHasVisibleDescendant(child, searchQuery));
}

function getLikelihoodLabel(probability: number | null | undefined) {
  if (typeof probability !== "number") {
    return "Unavailable";
  }

  if (probability >= 0.8) {
    return "High";
  }

  if (probability >= 0.6) {
    return "Medium";
  }

  return "Low";
}

function buildTreeGrid(
  root: TokenTreeNode,
  expandedNodeIds: string[],
  searchQuery: string,
): {
  heads: HeadSummary[];
  rows: GridRow[];
  totalVisibleNodes: number;
} {
  const expandedIds = new Set(expandedNodeIds);
  const visibleHeads = root.children.filter((child) =>
    nodeHasVisibleDescendant(child, searchQuery),
  );
  const rowsByDepth = new Map<number, GridNode[][]>();
  let maxDepth = 1;
  let totalVisibleNodes = 0;

  const getCellsForDepth = (depth: number) => {
    if (!rowsByDepth.has(depth)) {
      rowsByDepth.set(
        depth,
        Array.from({ length: visibleHeads.length }, () => [] as GridNode[]),
      );
    }

    return rowsByDepth.get(depth)!;
  };

  const visitNode = (
    node: TokenTreeNode,
    {
      headIndex,
      parentLabel,
    }: {
      headIndex: number;
      parentLabel: string;
    },
  ): number => {
    if (!nodeHasVisibleDescendant(node, searchQuery)) {
      return 0;
    }

    const hasVisibleChildren = node.children.some((child) =>
      nodeHasVisibleDescendant(child, searchQuery),
    );

    maxDepth = Math.max(maxDepth, node.depth);
    totalVisibleNodes += 1;

    const cells = getCellsForDepth(node.depth);
    cells[headIndex].push({
      childCount: node.children.length,
      hasVisibleChildren,
      headIndex,
      matchesSearch: nodeMatchesSearch(node, searchQuery),
      node,
      parentLabel,
    });

    let visibleCount = 1;

    if (hasVisibleChildren && expandedIds.has(node.id)) {
      for (const child of node.children) {
        visibleCount += visitNode(child, {
          headIndex,
          parentLabel: node.token,
        });
      }
    }

    return visibleCount;
  };

  const heads = visibleHeads.map((head, headIndex) => ({
    headIndex,
    node: head,
    visibleCount: visitNode(head, {
      headIndex,
      parentLabel: "Start",
    }),
  }));

  const rows = Array.from({ length: maxDepth }, (_, index) => {
    const depth = index + 1;

    return {
      cells:
        rowsByDepth.get(depth) ??
        Array.from({ length: heads.length }, () => [] as GridNode[]),
      depth,
      label: depth === 1 ? "Heads" : `Step ${depth}`,
    };
  });

  return {
    heads,
    rows,
    totalVisibleNodes,
  };
}

export function TokenBranchTree({
  root,
  selectedNodeId,
  expandedNodeIds,
  searchQuery,
  onSelectNode,
  onToggleNode,
}: TokenBranchTreeProps) {
  const [isDragging, setIsDragging] = useState(false);
  const clickSuppressedRef = useRef(false);
  const dragStateRef = useRef<DragState>({
    dragging: false,
    pointerId: null,
    startLeft: 0,
    startTop: 0,
    startX: 0,
    startY: 0,
  });
  const nodeRefs = useRef<Record<string, HTMLDivElement | null>>({});
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const { heads, rows, totalVisibleNodes } = buildTreeGrid(
    root,
    expandedNodeIds,
    searchQuery,
  );
  const headSummaryMap = new Map(heads.map((head) => [head.node.id, head.visibleCount]));

  useEffect(() => {
    if (!selectedNodeId) {
      return;
    }

    const selectedNode = nodeRefs.current[selectedNodeId];
    selectedNode?.scrollIntoView({
      behavior: "smooth",
      block: "center",
      inline: "center",
    });
  }, [selectedNodeId]);

  const finishDrag = (pointerId: number) => {
    const viewport = viewportRef.current;
    const state = dragStateRef.current;

    if (state.pointerId !== pointerId) {
      return;
    }

    if (viewport?.hasPointerCapture(pointerId)) {
      viewport.releasePointerCapture(pointerId);
    }

    dragStateRef.current = {
      dragging: false,
      pointerId: null,
      startLeft: 0,
      startTop: 0,
      startX: 0,
      startY: 0,
    };
    setIsDragging(false);
  };

  const handlePointerDown = (event: React.PointerEvent<HTMLDivElement>) => {
    if (event.pointerType === "mouse" && event.button !== 0) {
      return;
    }

    const target = event.target as HTMLElement;
    if (target.closest("[data-tree-ignore-drag='true']")) {
      return;
    }

    const viewport = viewportRef.current;

    if (!viewport) {
      return;
    }

    dragStateRef.current = {
      dragging: false,
      pointerId: event.pointerId,
      startLeft: viewport.scrollLeft,
      startTop: viewport.scrollTop,
      startX: event.clientX,
      startY: event.clientY,
    };

    viewport.setPointerCapture(event.pointerId);
  };

  const handlePointerMove = (event: React.PointerEvent<HTMLDivElement>) => {
    const viewport = viewportRef.current;
    const dragState = dragStateRef.current;

    if (!viewport || dragState.pointerId !== event.pointerId) {
      return;
    }

    const deltaX = event.clientX - dragState.startX;
    const deltaY = event.clientY - dragState.startY;
    const movedEnough = Math.abs(deltaX) > 5 || Math.abs(deltaY) > 5;

    if (!dragState.dragging && movedEnough) {
      dragStateRef.current.dragging = true;
      clickSuppressedRef.current = true;
      setIsDragging(true);
    }

    if (!dragStateRef.current.dragging) {
      return;
    }

    viewport.scrollLeft = dragState.startLeft - deltaX;
    viewport.scrollTop = dragState.startTop - deltaY;
    event.preventDefault();
  };

  const handlePointerUp = (event: React.PointerEvent<HTMLDivElement>) => {
    finishDrag(event.pointerId);
  };

  const handlePointerCancel = (event: React.PointerEvent<HTMLDivElement>) => {
    finishDrag(event.pointerId);
  };

  const handleNodeSelect = (nodeId: string) => {
    if (clickSuppressedRef.current) {
      clickSuppressedRef.current = false;
      return;
    }

    onSelectNode(nodeId);
  };

  if (heads.length === 0 || rows.length === 0) {
    return (
      <div className="flex min-h-[620px] items-center justify-center rounded-[2rem] border border-dashed border-slate-300 bg-white/70 px-6 text-sm text-slate-500">
        No visible branches.
      </div>
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center justify-between gap-3 rounded-2xl border border-slate-200 bg-white/85 px-4 py-3">
        <div className="inline-flex items-center gap-2 text-sm font-medium text-slate-600">
          <Move className="h-4 w-4 text-sky-600" />
          Drag to pan
        </div>
        <div className="flex flex-wrap items-center gap-3">
          <div className="flex items-center gap-2 text-xs font-medium uppercase tracking-[0.08em] text-slate-500">
            <span>Low</span>
            <div className="h-2 w-24 rounded-full bg-[linear-gradient(90deg,rgba(148,163,184,0.3),rgba(14,165,233,0.95))]" />
            <span>High</span>
          </div>
          <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium uppercase tracking-[0.08em] text-slate-600">
            {heads.length} heads
          </span>
          <span className="rounded-full border border-slate-200 bg-slate-50 px-3 py-2 text-xs font-medium uppercase tracking-[0.08em] text-slate-600">
            {totalVisibleNodes} visible nodes
          </span>
        </div>
      </div>

      <div
        ref={viewportRef}
        className={cn(
          "overflow-auto rounded-[2rem] border border-slate-200 bg-[radial-gradient(circle_at_top_left,_rgba(14,165,233,0.12),_transparent_32%),linear-gradient(to_bottom,_rgba(255,255,255,0.98),_rgba(241,245,249,0.96))]",
          isDragging ? "cursor-grabbing select-none" : "cursor-grab",
        )}
        onPointerCancel={handlePointerCancel}
        onPointerDown={handlePointerDown}
        onPointerMove={handlePointerMove}
        onPointerUp={handlePointerUp}
      >
        <div className="min-w-max p-6">
          <div
            className="grid auto-rows-auto gap-x-6 gap-y-5"
            style={{
              gridTemplateColumns: `120px repeat(${heads.length}, 320px)`,
            }}
          >
            <div className="rounded-2xl border border-slate-200 bg-white/90 px-4 py-3 shadow-sm" />

            {heads.map((head) => {
              const accent = HEAD_ACCENTS[head.headIndex % HEAD_ACCENTS.length];

              return (
                <div
                  key={`label-${head.node.id}`}
                  className="rounded-2xl border bg-white/90 px-4 py-3 shadow-sm"
                  style={{
                    borderColor: `rgb(${accent.rgb} / 0.22)`,
                  }}
                >
                  <div className="flex items-center justify-between gap-3">
                    <span
                      className={cn(
                        "rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.08em]",
                        accent.textClass,
                      )}
                      style={{
                        backgroundColor: `rgb(${accent.rgb} / 0.14)`,
                      }}
                    >
                      {accent.label}
                    </span>
                    <span className="text-xs font-medium uppercase tracking-[0.08em] text-slate-500">
                      {head.visibleCount} visible
                    </span>
                  </div>
                </div>
              );
            })}

            {rows.map((row) => (
              <div key={`row-${row.depth}`} className="contents">
                <div className="rounded-2xl border border-slate-200 bg-white/90 px-4 py-4 shadow-sm">
                  <p className="text-xs font-semibold uppercase tracking-[0.08em] text-slate-500">
                    {row.label}
                  </p>
                </div>

                {row.cells.map((cellNodes, headIndex) => {
                  const accent = HEAD_ACCENTS[headIndex % HEAD_ACCENTS.length];

                  return (
                    <div
                      key={`cell-${row.depth}-${headIndex}`}
                      className="min-h-[180px] rounded-[1.9rem] border p-3"
                      style={{
                        background: `linear-gradient(180deg, rgb(${accent.rgb} / 0.07), rgba(255,255,255,0.98))`,
                        borderColor: `rgb(${accent.rgb} / 0.18)`,
                      }}
                    >
                      {cellNodes.length > 0 ? (
                        <div className="space-y-3">
                          {cellNodes.map((entry) => {
                            const isExpanded = expandedNodeIds.includes(entry.node.id);
                            const isSelected = entry.node.id === selectedNodeId;
                            const localLikelihood = getLikelihoodLabel(entry.node.probability);
                            const cardGlow = 0.08 + (entry.node.cumulative_probability ?? 0) * 0.2;
                            const isHeadNode = entry.node.depth === 1;
                            const visibleCount = headSummaryMap.get(entry.node.id);

                            return (
                              <div
                                key={entry.node.id}
                                ref={(element) => {
                                  nodeRefs.current[entry.node.id] = element;
                                }}
                                className="rounded-[1.7rem] border p-4 transition-all"
                                style={{
                                  background: `linear-gradient(155deg, rgb(${accent.rgb} / ${cardGlow}), rgba(255,255,255,0.98))`,
                                  borderColor: isSelected
                                    ? `rgb(${accent.rgb} / 0.72)`
                                    : entry.matchesSearch
                                      ? `rgb(${accent.rgb} / 0.38)`
                                      : `rgb(${accent.rgb} / 0.2)`,
                                  boxShadow: isSelected
                                    ? `0 26px 58px -36px rgb(${accent.rgb} / 0.62)`
                                    : `0 18px 42px -34px rgb(${accent.rgb} / 0.22)`,
                                }}
                              >
                                <div className="flex items-start justify-between gap-3">
                                  <div className="min-w-0 space-y-2">
                                    <div className="flex flex-wrap items-center gap-2">
                                      <span className="truncate text-lg font-semibold text-slate-950">
                                        {entry.node.token}
                                      </span>
                                      <span
                                        className={cn(
                                          "rounded-full px-2.5 py-1 text-xs font-semibold uppercase tracking-[0.08em]",
                                          accent.textClass,
                                        )}
                                        style={{
                                          backgroundColor: `rgb(${accent.rgb} / 0.14)`,
                                        }}
                                      >
                                        {localLikelihood}
                                      </span>
                                      {entry.node.is_selected_path ? (
                                        <span className="rounded-full bg-white/85 px-2.5 py-1 text-xs font-medium text-slate-600">
                                          Main
                                        </span>
                                      ) : null}
                                    </div>

                                    <p className="text-xs font-medium uppercase tracking-[0.08em] text-slate-400">
                                      from {entry.parentLabel}
                                    </p>
                                  </div>

                                  <button
                                    className={cn(
                                      "inline-flex h-10 w-10 items-center justify-center rounded-full border bg-white/85 text-slate-600 shadow-sm",
                                      entry.hasVisibleChildren
                                        ? "border-white/80 hover:bg-white"
                                        : "cursor-default border-slate-100 opacity-40",
                                    )}
                                    data-tree-ignore-drag="true"
                                    disabled={!entry.hasVisibleChildren}
                                    onClick={() => onToggleNode(entry.node.id)}
                                    type="button"
                                  >
                                    {entry.hasVisibleChildren ? (
                                      isExpanded ? (
                                        <ChevronDown className="h-4 w-4" />
                                      ) : (
                                        <ChevronRight className="h-4 w-4" />
                                      )
                                    ) : (
                                      <ChevronRight className="h-4 w-4" />
                                    )}
                                  </button>
                                </div>

                                <div
                                  className={cn(
                                    "mt-4 rounded-[1.45rem] border border-white/80 bg-white/88 p-4 text-left transition-transform hover:-translate-y-0.5",
                                    isSelected ? "ring-2 ring-white/75" : "",
                                  )}
                                  onClick={() => handleNodeSelect(entry.node.id)}
                                  onKeyDown={(event) => {
                                    if (event.key === "Enter" || event.key === " ") {
                                      event.preventDefault();
                                      handleNodeSelect(entry.node.id);
                                    }
                                  }}
                                  role="button"
                                  tabIndex={0}
                                >
                                  <div className="space-y-3">
                                    <div className="space-y-1.5">
                                      <div className="flex items-center justify-between text-xs font-medium text-slate-600">
                                        <span>Local</span>
                                        <span>{formatPercent(entry.node.probability)}</span>
                                      </div>
                                      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                                        <div
                                          className="h-full rounded-full"
                                          style={{
                                            background: `linear-gradient(90deg, rgb(${accent.rgb} / 0.68), rgb(${accent.rgb}))`,
                                            width: `${Math.max((entry.node.probability ?? 0) * 100, 4)}%`,
                                          }}
                                        />
                                      </div>
                                    </div>

                                    <div className="space-y-1.5">
                                      <div className="flex items-center justify-between text-xs font-medium text-slate-600">
                                        <span>Path</span>
                                        <span>{formatPercent(entry.node.cumulative_probability)}</span>
                                      </div>
                                      <div className="h-2.5 overflow-hidden rounded-full bg-slate-100">
                                        <div
                                          className="h-full rounded-full"
                                          style={{
                                            background: `linear-gradient(90deg, rgb(${accent.rgb} / 0.32), rgb(${accent.rgb} / 0.82))`,
                                            width: `${Math.max((entry.node.cumulative_probability ?? 0) * 100, 3)}%`,
                                          }}
                                        />
                                      </div>
                                    </div>
                                  </div>

                                  <p className="mt-4 text-sm leading-7 text-slate-700">
                                    {entry.node.text_preview || entry.node.token}
                                  </p>

                                  <div className="mt-4 flex flex-wrap items-center gap-2 text-xs font-medium text-slate-500">
                                    <span className="rounded-full bg-slate-50 px-3 py-2">
                                      {entry.node.latency_ms} ms
                                    </span>
                                    {isHeadNode && typeof visibleCount === "number" ? (
                                      <span className="rounded-full bg-slate-50 px-3 py-2">
                                        {visibleCount} visible
                                      </span>
                                    ) : null}
                                    <span className="rounded-full bg-slate-50 px-3 py-2">
                                      {entry.childCount === 0
                                        ? "Leaf"
                                        : isExpanded
                                          ? `${entry.childCount} open`
                                          : `${entry.childCount} hidden`}
                                    </span>
                                  </div>
                                </div>
                              </div>
                            );
                          })}
                        </div>
                      ) : (
                        <div className="flex h-full min-h-[150px] items-center justify-center rounded-[1.45rem] border border-dashed border-slate-200 bg-white/65 text-xs font-medium uppercase tracking-[0.08em] text-slate-400">
                          Empty
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
