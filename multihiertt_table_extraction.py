import re
from typing import List, Tuple, Optional

class MultiHierttTableExtractor:
    """Extract tables from MultiHiertt dataset documents"""
    
    @staticmethod
    def has_table(text: str) -> bool:
        """Check if text contains a table"""
        lines = text.split('\n')
        pipe_lines = sum(1 for line in lines if '|' in line and line.strip())
        return pipe_lines >= 3
    
    @staticmethod
    def extract_table(text: str) -> Optional[Tuple[str, str]]:
        """
        Extract table from text
        Returns: (text_before_table, table_content) or None
        """
        if not MultiHierttTableExtractor.has_table(text):
            return None
        
        lines = text.split('\n')
        
        # Find table start
        table_start = None
        for i, line in enumerate(lines):
            if '|' in line and line.strip():
                pipe_count = sum(1 for j in range(i, min(i + 5, len(lines))) if '|' in lines[j])
                if pipe_count >= 3:
                    table_start = i
                    break
        
        if table_start is None:
            return None
        
        # Find table end
        table_end = table_start
        for i in range(table_start, len(lines)):
            line = lines[i].strip()
            if '|' in line or (line and all(c in '-| +' for c in line)):
                table_end = i
            else:
                if line:
                    break
        
        text_before = '\n'.join(lines[:table_start]).strip()
        table_content = '\n'.join(lines[table_start:table_end + 1]).strip()
        
        return text_before, table_content
    
    @staticmethod
    def parse_table_to_text(table_content: str) -> str:
        """
        Convert MultiHiertt table to readable text
        MultiHiertt has hierarchical headers:
        Row 1: Empty | Spanning header
        Row 2: Empty | Column headers (e.g., 2006, 2005)
        Row 3: Empty | Subheader (e.g., "(In millions)")
        Row 4+: Row label | Data values
        """
        lines = [line.strip() for line in table_content.split('\n') if line.strip()]
        
        # Filter out separator lines
        data_lines = []
        for line in lines:
            clean_line = line.replace('|', '').replace(' ', '').replace('+', '')
            if clean_line and not all(c == '-' for c in clean_line):
                data_lines.append(line)
        
        if len(data_lines) < 4:  # Need at least spanning header, column headers, subheader, and 1 data row
            return table_content
        
        # Parse all rows
        all_rows = [MultiHierttTableExtractor._parse_table_row(line) for line in data_lines]
        
        # Find the row with actual column headers (usually row with years like 2006, 2005)
        header_idx = 1  # Default to second row
        for i in range(min(3, len(all_rows))):
            row = all_rows[i]
            # Look for year-like numbers in the columns
            if len(row) > 1 and any(re.match(r'^\d{4}$', cell.strip()) for cell in row[1:] if cell.strip()):
                header_idx = i
                break
        
        # Get column headers (skip first empty column)
        headers = all_rows[header_idx]
        
        result = []
        
        # Process data rows (start after subheader row)
        start_idx = header_idx + 1
        # Skip rows that look like subheaders (contain only parenthetical info)
        while start_idx < len(all_rows):
            row = all_rows[start_idx]
            if len(row) > 1 and all(not cell or '(' in cell for cell in row[1:]):
                start_idx += 1
            else:
                break
        
        for i in range(start_idx, len(all_rows)):
            row = all_rows[i]
            
            if not row or len(row) < 2:
                continue
            
            # First cell is the row label
            row_label = row[0].strip()
            
            # Skip empty row labels
            if not row_label:
                continue
            
            # Ensure enough cells
            while len(row) < len(headers):
                row.append('')
            
            row_text = []
            # Start from index 1 (skip the first column which is the row label column)
            for j in range(1, len(headers)):
                if j < len(row):
                    header_value = headers[j].strip()
                    cell_value = row[j].strip()
                    
                    if header_value and cell_value:
                        row_text.append(f"{row_label} - {header_value}: {cell_value}")
            
            if row_text:
                result.append(" | ".join(row_text))
        
        return '\n'.join(result)
    
    @staticmethod
    def _parse_table_row(line: str) -> List[str]:
        """Parse a single table row"""
        cells = line.split('|')
        
        if cells and not cells[0].strip():
            cells = cells[1:]
        if cells and not cells[-1].strip():
            cells = cells[:-1]
        
        return [cell.strip() for cell in cells]
