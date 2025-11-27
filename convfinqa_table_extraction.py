import re
from typing import List, Tuple, Optional

class ConvFinQATableExtractor:
    """Extract tables from ConvFinQA dataset documents"""
    
    @staticmethod
    def has_table(text: str) -> bool:
        """Check if text contains a table"""
        lines = text.split('\n')
        pipe_lines = sum(1 for line in lines if '|' in line and line.strip())
        dash_lines = sum(1 for line in lines if line.strip().startswith('-'))
        return pipe_lines >= 3 or (pipe_lines >= 2 and dash_lines >= 1)
    
    @staticmethod
    def extract_table(text: str) -> Optional[Tuple[str, str]]:
        """
        Extract table from text
        Returns: (text_before_table, table_content) or None
        """
        if not ConvFinQATableExtractor.has_table(text):
            return None
        
        lines = text.split('\n')
        
        # Find table start
        table_start = None
        for i, line in enumerate(lines):
            if '|' in line and line.strip():
                # Look ahead for dash separator or more pipe lines
                has_separator = False
                pipe_count = 0
                for j in range(i, min(i + 5, len(lines))):
                    if '|' in lines[j]:
                        pipe_count += 1
                    if lines[j].strip().startswith('-'):
                        has_separator = True
                
                if pipe_count >= 2 or has_separator:
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
        Convert ConvFinQA table to readable text
        ConvFinQA format: Header row, dash separator, then data rows
        """
        lines = [line.strip() for line in table_content.split('\n') if line.strip()]
        
        # Filter out separator lines (lines with mostly dashes)
        data_lines = []
        for line in lines:
            clean_line = line.replace('|', '').replace(' ', '').replace('+', '')
            if clean_line and not all(c == '-' for c in clean_line):
                data_lines.append(line)
        
        if len(data_lines) < 2:
            return table_content
        
        # First line is header
        headers = ConvFinQATableExtractor._parse_table_row(data_lines[0])
        
        result = []
        
        # Process data rows
        for row_line in data_lines[1:]:
            cells = ConvFinQATableExtractor._parse_table_row(row_line)
            
            # Ensure enough cells
            while len(cells) < len(headers):
                cells.append('')
            
            row_text = []
            for header, cell in zip(headers, cells):
                cell_value = cell.strip()
                header_value = header.strip()
                
                if header_value and cell_value:
                    row_text.append(f"{header_value}: {cell_value}")
            
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
