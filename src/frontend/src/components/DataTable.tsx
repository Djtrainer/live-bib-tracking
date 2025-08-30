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
    <div className="overflow-x-auto">
      {/* For best results, ensure this table has a fixed layout and full width */}
      <table className="w-full table-fixed border-collapse">
        <thead>
          <tr>
            {columns.map((column) => (
              <th 
                key={column.key} 
                style={{ width: column.width }}
                // Use template literals to construct the className for alignment
                className={`text-${column.align || 'left'} px-4 py-3 font-medium text-muted-foreground`}
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
            // --- FIX: Pass the 'columns' array to the renderRow function ---
            data.map((item, index) => renderRow(item, index, columns))
          )}
        </tbody>
      </table>
    </div>
  );
}