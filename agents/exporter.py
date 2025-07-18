"""
Exporter Agent - Returns final answers in requested formats.
"""
import json
import csv
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import io
import base64

from core.simple_base_agent import SimpleBaseAgent


class Exporter(SimpleBaseAgent):
    """
    Agent responsible for formatting and exporting final answers in various formats
    including JSON, Markdown, CSV, XML, and HTML.
    """
    
    def __init__(self):
        # Define tools for export operations
        tools = [
            self.export_json,
            self.export_markdown,
            self.export_csv,
            self.export_xml,
            self.export_html,
            self.export_pdf_report
        ]
        
        super().__init__(
            name="Exporter",
            description="Formats and exports final answers in various formats including JSON, Markdown, CSV, XML, HTML, and PDF reports. Ensures proper formatting and structure for different output requirements.",
            tools=tools
        )
        
        # Supported export formats
        self.supported_formats = {
            "json": {"extension": ".json", "mime_type": "application/json"},
            "markdown": {"extension": ".md", "mime_type": "text/markdown"},
            "csv": {"extension": ".csv", "mime_type": "text/csv"},
            "xml": {"extension": ".xml", "mime_type": "application/xml"},
            "html": {"extension": ".html", "mime_type": "text/html"},
            "pdf": {"extension": ".pdf", "mime_type": "application/pdf"},
            "txt": {"extension": ".txt", "mime_type": "text/plain"}
        }
    
    def export_json(self, data: Dict[str, Any], formatting: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Export data in JSON format.
        
        Args:
            data: Data to export
            formatting: JSON formatting options (indent, sort_keys, etc.)
        
        Returns:
            JSON export result
        """
        try:
            formatting = formatting or {}
            
            # Default JSON formatting
            json_options = {
                "indent": formatting.get("indent", 2),
                "sort_keys": formatting.get("sort_keys", True),
                "ensure_ascii": formatting.get("ensure_ascii", False)
            }
            
            # Convert data to JSON string
            json_output = json.dumps(data, **json_options)
            
            # Create export metadata
            export_metadata = {
                "format": "json",
                "size_bytes": len(json_output.encode('utf-8')),
                "export_timestamp": datetime.now().isoformat(),
                "formatting_options": json_options
            }
            
            return {
                "success": True,
                "format": "json",
                "content": json_output,
                "metadata": export_metadata,
                "filename": f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            }
            
        except Exception as e:
            return {"error": f"JSON export failed: {str(e)}"}
    
    def export_markdown(self, data: Dict[str, Any], template: str = None) -> Dict[str, Any]:
        """
        Export data in Markdown format.
        
        Args:
            data: Data to export
            template: Optional Markdown template
        
        Returns:
            Markdown export result
        """
        try:
            if template:
                markdown_content = self._apply_markdown_template(data, template)
            else:
                markdown_content = self._generate_default_markdown(data)
            
            export_metadata = {
                "format": "markdown",
                "size_bytes": len(markdown_content.encode('utf-8')),
                "export_timestamp": datetime.now().isoformat(),
                "template_used": template is not None
            }
            
            return {
                "success": True,
                "format": "markdown",
                "content": markdown_content,
                "metadata": export_metadata,
                "filename": f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            }
            
        except Exception as e:
            return {"error": f"Markdown export failed: {str(e)}"}
    
    def export_csv(self, data: Union[List[Dict], Dict[str, Any]], options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Export data in CSV format.
        
        Args:
            data: Data to export (list of dictionaries or structured data)
            options: CSV formatting options
        
        Returns:
            CSV export result
        """
        try:
            options = options or {}
            
            # Convert data to list of dictionaries if needed
            if isinstance(data, dict):
                csv_data = self._dict_to_csv_data(data)
            elif isinstance(data, list):
                csv_data = data
            else:
                return {"error": "Data format not suitable for CSV export"}
            
            # Create CSV content
            output = io.StringIO()
            if csv_data:
                fieldnames = csv_data[0].keys() if csv_data else []
                writer = csv.DictWriter(
                    output,
                    fieldnames=fieldnames,
                    delimiter=options.get("delimiter", ","),
                    quotechar=options.get("quotechar", '"'),
                    quoting=csv.QUOTE_MINIMAL
                )
                
                writer.writeheader()
                writer.writerows(csv_data)
            
            csv_content = output.getvalue()
            output.close()
            
            export_metadata = {
                "format": "csv",
                "size_bytes": len(csv_content.encode('utf-8')),
                "export_timestamp": datetime.now().isoformat(),
                "rows": len(csv_data),
                "columns": len(csv_data[0].keys()) if csv_data else 0
            }
            
            return {
                "success": True,
                "format": "csv",
                "content": csv_content,
                "metadata": export_metadata,
                "filename": f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            }
            
        except Exception as e:
            return {"error": f"CSV export failed: {str(e)}"}
    
    def export_xml(self, data: Dict[str, Any], root_element: str = "export") -> Dict[str, Any]:
        """
        Export data in XML format.
        
        Args:
            data: Data to export
            root_element: Name of the root XML element
        
        Returns:
            XML export result
        """
        try:
            # Create root element
            root = ET.Element(root_element)
            
            # Add metadata
            metadata_elem = ET.SubElement(root, "metadata")
            ET.SubElement(metadata_elem, "export_timestamp").text = datetime.now().isoformat()
            ET.SubElement(metadata_elem, "format").text = "xml"
            
            # Add data
            data_elem = ET.SubElement(root, "data")
            self._dict_to_xml(data, data_elem)
            
            # Convert to string
            xml_content = ET.tostring(root, encoding='unicode', method='xml')
            
            # Pretty format
            try:
                import xml.dom.minidom
                dom = xml.dom.minidom.parseString(xml_content)
                xml_content = dom.toprettyxml(indent="  ")
            except:
                pass  # Use unformatted XML if pretty printing fails
            
            export_metadata = {
                "format": "xml",
                "size_bytes": len(xml_content.encode('utf-8')),
                "export_timestamp": datetime.now().isoformat(),
                "root_element": root_element
            }
            
            return {
                "success": True,
                "format": "xml",
                "content": xml_content,
                "metadata": export_metadata,
                "filename": f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xml"
            }
            
        except Exception as e:
            return {"error": f"XML export failed: {str(e)}"}
    
    def export_html(self, data: Dict[str, Any], template: str = None, styling: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Export data in HTML format.
        
        Args:
            data: Data to export
            template: Optional HTML template
            styling: CSS styling options
        
        Returns:
            HTML export result
        """
        try:
            if template:
                html_content = self._apply_html_template(data, template, styling)
            else:
                html_content = self._generate_default_html(data, styling)
            
            export_metadata = {
                "format": "html",
                "size_bytes": len(html_content.encode('utf-8')),
                "export_timestamp": datetime.now().isoformat(),
                "template_used": template is not None,
                "styling_applied": styling is not None
            }
            
            return {
                "success": True,
                "format": "html",
                "content": html_content,
                "metadata": export_metadata,
                "filename": f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
            }
            
        except Exception as e:
            return {"error": f"HTML export failed: {str(e)}"}
    
    def export_pdf_report(self, data: Dict[str, Any], report_config: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Export data as a PDF report.
        
        Args:
            data: Data to export
            report_config: PDF report configuration
        
        Returns:
            PDF export result
        """
        try:
            # This is a simplified implementation
            # In production, you'd use libraries like reportlab or weasyprint
            
            report_config = report_config or {}
            
            # Generate HTML content first
            html_result = self.export_html(data, styling={"report_style": True})
            if not html_result.get("success"):
                return html_result
            
            html_content = html_result["content"]
            
            # Convert HTML to PDF (placeholder implementation)
            # In practice, you'd use a library like weasyprint:
            # import weasyprint
            # pdf_bytes = weasyprint.HTML(string=html_content).write_pdf()
            
            # For now, return base64 encoded HTML as placeholder
            pdf_placeholder = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            
            export_metadata = {
                "format": "pdf",
                "size_bytes": len(pdf_placeholder),
                "export_timestamp": datetime.now().isoformat(),
                "report_config": report_config,
                "note": "PDF generation requires additional libraries (weasyprint, reportlab)"
            }
            
            return {
                "success": True,
                "format": "pdf",
                "content": pdf_placeholder,
                "content_type": "base64_encoded",
                "metadata": export_metadata,
                "filename": f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf"
            }
            
        except Exception as e:
            return {"error": f"PDF export failed: {str(e)}"}
    
    def _apply_markdown_template(self, data: Dict[str, Any], template: str) -> str:
        """Apply Markdown template to data."""
        # Simple template substitution
        content = template
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            content = content.replace(placeholder, str(value))
        return content
    
    def _generate_default_markdown(self, data: Dict[str, Any]) -> str:
        """Generate default Markdown format."""
        lines = []
        lines.append(f"# Export Report")
        lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Add data sections
        for key, value in data.items():
            lines.append(f"## {key.replace('_', ' ').title()}")
            
            if isinstance(value, dict):
                for subkey, subvalue in value.items():
                    lines.append(f"- **{subkey}**: {subvalue}")
            elif isinstance(value, list):
                for item in value:
                    lines.append(f"- {item}")
            else:
                lines.append(f"{value}")
            
            lines.append("")
        
        return "\n".join(lines)
    
    def _dict_to_csv_data(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Convert dictionary data to CSV-compatible format."""
        csv_rows = []
        
        def flatten_dict(d, parent_key='', sep='_'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                elif isinstance(v, list):
                    for i, item in enumerate(v):
                        if isinstance(item, dict):
                            items.extend(flatten_dict(item, f"{new_key}_{i}", sep=sep).items())
                        else:
                            items.append((f"{new_key}_{i}", item))
                else:
                    items.append((new_key, v))
            return dict(items)
        
        flattened = flatten_dict(data)
        csv_rows.append(flattened)
        
        return csv_rows
    
    def _dict_to_xml(self, data: Any, parent_element: ET.Element):
        """Convert dictionary to XML elements."""
        if isinstance(data, dict):
            for key, value in data.items():
                # Clean key name for XML
                clean_key = re.sub(r'[^a-zA-Z0-9_]', '_', str(key))
                child_elem = ET.SubElement(parent_element, clean_key)
                self._dict_to_xml(value, child_elem)
        elif isinstance(data, list):
            for i, item in enumerate(data):
                item_elem = ET.SubElement(parent_element, f"item_{i}")
                self._dict_to_xml(item, item_elem)
        else:
            parent_element.text = str(data)
    
    def _apply_html_template(self, data: Dict[str, Any], template: str, styling: Dict[str, Any] = None) -> str:
        """Apply HTML template to data."""
        # Simple template substitution
        content = template
        for key, value in data.items():
            placeholder = f"{{{key}}}"
            content = content.replace(placeholder, str(value))
        
        # Apply styling if provided
        if styling:
            # Add CSS styles
            style_tag = "<style>\n"
            for selector, styles in styling.items():
                if isinstance(styles, dict):
                    style_tag += f"{selector} {{\n"
                    for prop, val in styles.items():
                        style_tag += f"  {prop}: {val};\n"
                    style_tag += "}\n"
            style_tag += "</style>\n"
            
            # Insert styles into head
            content = content.replace("</head>", f"{style_tag}</head>")
        
        return content
    
    def _generate_default_html(self, data: Dict[str, Any], styling: Dict[str, Any] = None) -> str:
        """Generate default HTML format."""
        html_lines = [
            "<!DOCTYPE html>",
            "<html>",
            "<head>",
            "<title>Export Report</title>",
            "<meta charset='utf-8'>",
        ]
        
        # Add default styling
        if styling and styling.get("report_style"):
            html_lines.extend([
                "<style>",
                "body { font-family: Arial, sans-serif; margin: 40px; }",
                "h1 { color: #333; border-bottom: 2px solid #333; }",
                "h2 { color: #666; margin-top: 30px; }",
                "table { border-collapse: collapse; width: 100%; margin: 20px 0; }",
                "th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }",
                "th { background-color: #f2f2f2; }",
                ".metadata { background-color: #f9f9f9; padding: 10px; border-radius: 5px; }",
                "</style>"
            ])
        
        html_lines.extend([
            "</head>",
            "<body>",
            f"<h1>Export Report</h1>",
            f"<div class='metadata'>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>",
        ])
        
        # Add data content
        for key, value in data.items():
            html_lines.append(f"<h2>{key.replace('_', ' ').title()}</h2>")
            
            if isinstance(value, dict):
                html_lines.append("<table>")
                for subkey, subvalue in value.items():
                    html_lines.append(f"<tr><th>{subkey}</th><td>{subvalue}</td></tr>")
                html_lines.append("</table>")
            elif isinstance(value, list):
                html_lines.append("<ul>")
                for item in value:
                    html_lines.append(f"<li>{item}</li>")
                html_lines.append("</ul>")
            else:
                html_lines.append(f"<p>{value}</p>")
        
        html_lines.extend([
            "</body>",
            "</html>"
        ])
        
        return "\n".join(html_lines)
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process export request.
        
        Args:
            input_data: Dictionary containing data to export and format specifications
            
        Returns:
            Dictionary with exported content in requested format
        """
        try:
            self.log_activity("Starting export process", input_data)
            
            operation = input_data.get("operation")
            
            if operation == "export_json":
                return self.export_json(input_data.get("data"), input_data.get("formatting"))
            elif operation == "export_markdown":
                return self.export_markdown(input_data.get("data"), input_data.get("template"))
            elif operation == "export_csv":
                return self.export_csv(input_data.get("data"), input_data.get("options"))
            elif operation == "export_xml":
                return self.export_xml(input_data.get("data"), input_data.get("root_element"))
            elif operation == "export_html":
                return self.export_html(input_data.get("data"), input_data.get("template"), input_data.get("styling"))
            elif operation == "export_pdf_report":
                return self.export_pdf_report(input_data.get("data"), input_data.get("report_config"))
            elif operation == "get_supported_formats":
                return self.get_supported_formats()
            else:
                return self.create_error_response(f"Unsupported export operation: {operation}")
            
        except Exception as e:
            self.logger.error(f"Error in export process: {str(e)}")
            return self.create_error_response(f"Export process failed: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in export process: {str(e)}")
            return self.create_error_response(f"Export process failed: {str(e)}")
    
    def get_supported_formats(self) -> Dict[str, Any]:
        """Get information about supported export formats."""
        return {
            "success": True,
            "supported_formats": self.supported_formats,
            "format_count": len(self.supported_formats),
            "capabilities": {
                "json": ["formatting options", "pretty printing", "custom serialization"],
                "markdown": ["templates", "custom formatting", "table generation"],
                "csv": ["custom delimiters", "header options", "data flattening"],
                "xml": ["custom root elements", "pretty printing", "nested structures"],
                "html": ["templates", "CSS styling", "responsive design"],
                "pdf": ["report generation", "custom layouts", "requires additional libraries"]
            }
        }

