import { ReactNode } from 'react';

interface Column {
  key: string;
  title: string;
  width?: string;
  align?: 'left' | 'center' | 'right';
}

interface DataTableProps {
  columns: Column[];
  data: any[];
  renderRow: (item: any, index: number, columns: Column[]) => ReactNode;
  emptyMessage?: string;
}

export default function DataTable({ columns, data, renderRow, emptyMessage = "No data available" }: DataTableProps) {
  return (
    <div className="overflow-hidden rounded-lg border border-border bg-card">
      <div className="overflow-x-auto">
        <table className="data-table w-full table-fixed">
          <thead>
            <tr>
              {columns.map((column) => (
                <th 
                  key={column.key} 
                  style={{ width: column.width }}
                  className={`text-${column.align || 'left'} px-4 py-4 font-semibold text-sm text-muted-foreground uppercase tracking-wider`}
                >
                  {column.title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="text-center py-12">
                  <div className="flex flex-col items-center gap-3">
                    <span className="material-icon text-4xl text-muted-foreground opacity-50">inbox</span>
                    <p className="text-muted-foreground font-medium">{emptyMessage}</p>
                  </div>
                </td>
              </tr>
            ) : (
              data.map((item, index) => renderRow(item, index, columns))
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}