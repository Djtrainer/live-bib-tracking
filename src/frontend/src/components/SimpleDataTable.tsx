import { ReactNode } from 'react';

interface Column {
  key: string;
  title: string;
  width?: string;
  align?: 'left' | 'center' | 'right';
}

interface SimpleDataTableProps {
  columns: Column[];
  data: any[];
  renderRow: (item: any, index: number, columns?: Column[]) => ReactNode;
  emptyMessage?: string;
}

export default function SimpleDataTable({ columns, data, renderRow, emptyMessage = "No data available" }: SimpleDataTableProps) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full table-fixed border-collapse">
        <thead>
          <tr className="bg-muted/30">
            {columns.map((column) => (
              <th 
                key={column.key} 
                style={{ width: column.width }}
                className={`text-${column.align || 'left'} px-4 py-3 font-semibold text-muted-foreground border-b-2 border-border`}
              >
                {column.title}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {data.length === 0 ? (
            <tr>
              <td colSpan={columns.length} className="text-center py-8 text-muted-foreground">
                {emptyMessage}
              </td>
            </tr>
          ) : (
            data.map((item, index) => renderRow(item, index, columns))
          )}
        </tbody>
      </table>
    </div>
  );
}