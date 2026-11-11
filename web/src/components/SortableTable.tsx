import { useMemo, useState, type ReactNode } from "react";

export interface Column<T> {
  id: string;
  label: string;
  align?: "left" | "right";
  getValue: (row: T) => string | number | null | undefined;
  sortValue?: (row: T) => number | string;
  render?: (row: T) => ReactNode;
}

type SortDir = "asc" | "desc";

interface Props<T> {
  rows: T[];
  columns: Column<T>[];
  defaultSortId: string;
  defaultDir?: SortDir;
  rowKey: (row: T, index: number) => string | number;
}

export function SortableTable<T>({
  rows,
  columns,
  defaultSortId,
  defaultDir = "asc",
  rowKey,
}: Props<T>) {
  const [sortId, setSortId] = useState(defaultSortId);
  const [sortDir, setSortDir] = useState<SortDir>(defaultDir);

  const sorted = useMemo(() => {
    const col = columns.find((c) => c.id === sortId);
    if (!col) return rows;
    const getter = col.sortValue ?? col.getValue;
    const copy = [...rows];
    copy.sort((a, b) => {
      const av = getter(a);
      const bv = getter(b);
      if (av == null && bv == null) return 0;
      if (av == null) return 1;
      if (bv == null) return -1;
      if (typeof av === "number" && typeof bv === "number") {
        return sortDir === "asc" ? av - bv : bv - av;
      }
      const as = String(av);
      const bs = String(bv);
      const cmp = as.localeCompare(bs, undefined, { numeric: true });
      return sortDir === "asc" ? cmp : -cmp;
    });
    return copy;
  }, [rows, columns, sortId, sortDir]);

  function onHeaderClick(id: string) {
    if (sortId === id) {
      setSortDir((d) => (d === "asc" ? "desc" : "asc"));
    } else {
      setSortId(id);
      setSortDir(id.endsWith("rank") || id === "rank" ? "asc" : "desc");
    }
  }

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {columns.map((col) => {
              const active = sortId === col.id;
              const arrow = active ? (sortDir === "asc" ? " ↑" : " ↓") : "";
              return (
                <th
                  key={col.id}
                  className={col.align === "right" ? "num" : undefined}
                  aria-sort={
                    active
                      ? sortDir === "asc"
                        ? "ascending"
                        : "descending"
                      : undefined
                  }
                >
                  <button
                    type="button"
                    className="th-btn"
                    onClick={() => onHeaderClick(col.id)}
                  >
                    {col.label}
                    {arrow}
                  </button>
                </th>
              );
            })}
          </tr>
        </thead>
        <tbody>
          {sorted.map((row, i) => (
            <tr key={rowKey(row, i)}>
              {columns.map((col) => (
                <td
                  key={col.id}
                  className={col.align === "right" ? "num" : undefined}
                >
                  {col.render
                    ? col.render(row)
                    : formatCell(col.getValue(row))}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value: string | number | null | undefined): string {
  if (value == null || value === "") return "—";
  if (typeof value === "number") {
    return Number.isInteger(value) ? String(value) : value.toFixed(1);
  }
  return String(value);
}
