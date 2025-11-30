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
        Convert MultiHiertt table to readable Markdown (.md) text
        MultiHiertt has hierarchical headers:
        
        Row 1: Empty | Spanning header\n
        Row 2: Empty | Column headers (e.g., 2006, 2005)\n
        Row 3: Empty | Subheader (e.g., "(In millions)")\n
        Row 4+: Row label | Data values
        """
        lines = [line.strip() for line in table_content.strip().split("\n") if line.strip()]

        # Parse rows into lists
        rows = []
        for line in lines:
            parts = [col.strip() for col in line.split("|")[1:-1]]
            if parts != []:
                rows.append(parts)

        # Detect "header metadata" rows — those where column 0 is empty
        meta_rows = [r for r in rows if r[0] == ""]
        data_rows = [r for r in rows if r[0] != ""]

        # Build column names dynamically
        # Example:
        # meta rows:
        #   ["", "Years Ended December 31,"]
        #   ["", "2006", "2005"]
        #   ["", "(In millions)"]
        num_columns = max(len(r) for r in meta_rows + data_rows) # search for the max columns

        # Initialize header labels
        headers = []

        # Combine metadata rows by column index
        col_labels = []        
        for meta in meta_rows:
            parts = []
            if len(meta) < num_columns:
                # duplicate the 
                temp_parts = meta[1:] * ((num_columns -1) // (len(meta)-1))
                parts = [""] + temp_parts
            else:
                parts = meta
            col_labels.append(parts)

        headers = col_labels
        markdown = MultiHierttTableExtractor._parse_to_markdown(headers, data_rows)
        return markdown
    
    @staticmethod
    def _parse_to_markdown(headers, data):
        """Function for parsing extracted tables items into markdown format

        Args:
            headers (list): Nested list of header strings
            data (list): Nested list of data row strings

        Returns:
            str: completed markdown string
        """
        md = ""
        first_header = True
        for header in headers:
            md += "|" + "| ".join(header) + " |\n"
            if first_header:
                md += "|" + "| ".join(["---"] * len(headers[0])) + " |\n"
                first_header = False
        
        for row in data:
            md += "|" + "| ".join(row) + " |\n"
        return md
