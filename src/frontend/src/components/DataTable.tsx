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
    <div className="overflow-hidden rounded-xl border border-border">
      <div className="overflow-x-auto">
        <table className="professional-table">
          <thead>
            <tr>
              {columns.map((column) => (
                <th 
                  key={column.key} 
                  style={{ width: column.width }}
                  className={`text-${column.align || 'left'} px-6 py-4 font-semibold text-sm tracking-wide`}
                >
                  {column.title}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {data.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="text-center py-16 text-muted-foreground">
                  <div className="flex flex-col items-center gap-3">
                    <span className="material-icon text-4xl opacity-50">inbox</span>
                    <div className="text-lg font-medium">{emptyMessage}</div>
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