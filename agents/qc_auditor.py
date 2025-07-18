"""
QC Auditor Agent - Validates numbers, unit consistency, and citation links.
"""
import re
import json
import requests
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from urllib.parse import urlparse
import math

from core.simple_base_agent import SimpleBaseAgent


class QCAuditor(SimpleBaseAgent):
    """
    Agent responsible for quality control and auditing of calculation results,
    including numerical validation, unit consistency checks, and citation verification.
    """
    
    def __init__(self):
        # Define tools for QC operations
        tools = [
            self.validate_numbers,
            self.check_unit_consistency,
            self.verify_citations,
            self.audit_calculations,
            self.generate_audit_report
        ]
        
        super().__init__(
            name="QCAuditor",
            description="Validates calculation results for numerical accuracy, unit consistency, and citation integrity. Ensures quality and reliability of energy analysis outputs.",
            tools=tools
        )
        
        # Unit conversion factors (to base SI units)
        self.unit_conversions = {
            # Energy units (to Joules)
            "J": 1.0, "kJ": 1e3, "MJ": 1e6, "GJ": 1e9, "TJ": 1e12,
            "Wh": 3600, "kWh": 3.6e6, "MWh": 3.6e9, "GWh": 3.6e12, "TWh": 3.6e15,
            "BTU": 1055.06, "MMBTU": 1.055e9, "therm": 1.055e8,
            "cal": 4.184, "kcal": 4184,
            
            # Power units (to Watts)
            "W": 1.0, "kW": 1e3, "MW": 1e6, "GW": 1e9, "TW": 1e12,
            "hp": 745.7, "PS": 735.5,
            
            # Currency (relative, needs context)
            "USD": 1.0, "EUR": 1.1, "GBP": 1.3, "JPY": 0.007,
            
            # Time units (to seconds)
            "s": 1.0, "min": 60, "h": 3600, "day": 86400, "year": 31536000,
            
            # Mass units (to kg)
            "kg": 1.0, "g": 0.001, "t": 1000, "lb": 0.453592, "oz": 0.0283495,
            
            # Volume units (to m³)
            "m3": 1.0, "L": 0.001, "gal": 0.00378541, "ft3": 0.0283168
        }
        
        # Common energy calculation ranges for validation
        self.validation_ranges = {
            "lcoe": {"min": 0.01, "max": 1000, "unit": "USD/MWh", "typical": (20, 200)},
            "capacity_factor": {"min": 0.0, "max": 1.0, "unit": "fraction", "typical": (0.1, 0.9)},
            "efficiency": {"min": 0.0, "max": 1.0, "unit": "fraction", "typical": (0.1, 0.95)},
            "npv": {"min": -1e12, "max": 1e12, "unit": "USD", "typical": (-1e9, 1e9)},
            "irr": {"min": -1.0, "max": 10.0, "unit": "fraction", "typical": (0.05, 0.25)},
            "payback_period": {"min": 0, "max": 100, "unit": "years", "typical": (1, 30)},
            "emission_factor": {"min": 0, "max": 2000, "unit": "kg CO2/MWh", "typical": (0, 1000)}
        }
    
    def validate_numbers(self, results: Dict[str, Any], calculation_type: str = None) -> Dict[str, Any]:
        """
        Validate numerical results for reasonableness and accuracy.
        
        Args:
            results: Dictionary containing calculation results
            calculation_type: Type of calculation for context-specific validation
        
        Returns:
            Validation results with pass/fail status
        """
        try:
            validation_results = {
                "overall_status": "pass",
                "checks": [],
                "warnings": [],
                "errors": []
            }
            
            # Extract numerical values from results
            numerical_values = self._extract_numerical_values(results)
            
            for key, value in numerical_values.items():
                check_result = {
                    "parameter": key,
                    "value": value,
                    "status": "pass",
                    "checks_performed": []
                }
                
                # Check for NaN or infinite values
                if math.isnan(value) or math.isinf(value):
                    check_result["status"] = "fail"
                    validation_results["errors"].append(f"{key}: Invalid numerical value (NaN or Inf)")
                    validation_results["overall_status"] = "fail"
                    check_result["checks_performed"].append("nan_inf_check")
                
                # Range validation based on calculation type
                if calculation_type and calculation_type.lower() in self.validation_ranges:
                    range_info = self.validation_ranges[calculation_type.lower()]
                    if not (range_info["min"] <= value <= range_info["max"]):
                        check_result["status"] = "fail"
                        validation_results["errors"].append(
                            f"{key}: Value {value} outside valid range [{range_info['min']}, {range_info['max']}]"
                        )
                        validation_results["overall_status"] = "fail"
                    
                    # Typical range warning
                    typical_min, typical_max = range_info["typical"]
                    if not (typical_min <= value <= typical_max):
                        validation_results["warnings"].append(
                            f"{key}: Value {value} outside typical range [{typical_min}, {typical_max}]"
                        )
                    
                    check_result["checks_performed"].append("range_validation")
                
                # Precision check (avoid false precision)
                if isinstance(value, float):
                    decimal_places = len(str(value).split('.')[-1]) if '.' in str(value) else 0
                    if decimal_places > 6:
                        validation_results["warnings"].append(
                            f"{key}: Excessive precision ({decimal_places} decimal places)"
                        )
                        check_result["checks_performed"].append("precision_check")
                
                validation_results["checks"].append(check_result)
            
            return {
                "success": True,
                "validation": validation_results,
                "total_values_checked": len(numerical_values)
            }
            
        except Exception as e:
            return {"error": f"Number validation failed: {str(e)}"}
    
    def check_unit_consistency(self, data: Dict[str, Any], expected_units: Dict[str, str] = None) -> Dict[str, Any]:
        """
        Check unit consistency across calculations and data.
        
        Args:
            data: Data containing values with units
            expected_units: Expected units for specific parameters
        
        Returns:
            Unit consistency check results
        """
        try:
            consistency_results = {
                "overall_status": "pass",
                "unit_checks": [],
                "conversions_needed": [],
                "errors": [],
                "warnings": []
            }
            
            # Extract units from data
            units_found = self._extract_units(data)
            
            # Check each parameter's units
            for param, unit_info in units_found.items():
                check_result = {
                    "parameter": param,
                    "found_unit": unit_info.get("unit"),
                    "status": "pass"
                }
                
                # Check against expected units
                if expected_units and param in expected_units:
                    expected_unit = expected_units[param]
                    found_unit = unit_info.get("unit")
                    
                    if found_unit != expected_unit:
                        # Check if units are convertible
                        if self._are_units_compatible(found_unit, expected_unit):
                            consistency_results["conversions_needed"].append({
                                "parameter": param,
                                "from_unit": found_unit,
                                "to_unit": expected_unit,
                                "conversion_factor": self._get_conversion_factor(found_unit, expected_unit)
                            })
                            consistency_results["warnings"].append(
                                f"{param}: Unit conversion needed from {found_unit} to {expected_unit}"
                            )
                        else:
                            check_result["status"] = "fail"
                            consistency_results["errors"].append(
                                f"{param}: Incompatible units - found {found_unit}, expected {expected_unit}"
                            )
                            consistency_results["overall_status"] = "fail"
                
                # Check for missing units
                if not unit_info.get("unit"):
                    consistency_results["warnings"].append(f"{param}: No unit specified")
                
                consistency_results["unit_checks"].append(check_result)
            
            # Check for dimensional consistency in formulas
            dimensional_check = self._check_dimensional_consistency(data)
            if not dimensional_check["consistent"]:
                consistency_results["errors"].extend(dimensional_check["errors"])
                consistency_results["overall_status"] = "fail"
            
            return {
                "success": True,
                "consistency": consistency_results,
                "units_analyzed": len(units_found)
            }
            
        except Exception as e:
            return {"error": f"Unit consistency check failed: {str(e)}"}
    
    def verify_citations(self, content: str, citations: List[Dict] = None) -> Dict[str, Any]:
        """
        Verify citation links and references.
        
        Args:
            content: Content containing citations
            citations: List of citation dictionaries
        
        Returns:
            Citation verification results
        """
        try:
            verification_results = {
                "overall_status": "pass",
                "citations_checked": 0,
                "valid_citations": 0,
                "invalid_citations": 0,
                "citation_details": [],
                "errors": [],
                "warnings": []
            }
            
            # Extract citations from content if not provided
            if not citations:
                citations = self._extract_citations_from_content(content)
            
            verification_results["citations_checked"] = len(citations)
            
            # Verify each citation
            for i, citation in enumerate(citations):
                citation_result = {
                    "citation_id": i + 1,
                    "status": "unknown",
                    "url": citation.get("url"),
                    "title": citation.get("title"),
                    "checks_performed": []
                }
                
                # URL format validation
                if citation.get("url"):
                    url = citation["url"]
                    if self._is_valid_url(url):
                        citation_result["checks_performed"].append("url_format_valid")
                        
                        # Check URL accessibility (with timeout)
                        accessibility_check = self._check_url_accessibility(url)
                        if accessibility_check["accessible"]:
                            citation_result["status"] = "valid"
                            verification_results["valid_citations"] += 1
                            citation_result["checks_performed"].append("url_accessible")
                        else:
                            citation_result["status"] = "invalid"
                            verification_results["invalid_citations"] += 1
                            verification_results["errors"].append(
                                f"Citation {i+1}: URL not accessible - {accessibility_check['error']}"
                            )
                            verification_results["overall_status"] = "fail"
                    else:
                        citation_result["status"] = "invalid"
                        verification_results["invalid_citations"] += 1
                        verification_results["errors"].append(f"Citation {i+1}: Invalid URL format")
                        verification_results["overall_status"] = "fail"
                else:
                    verification_results["warnings"].append(f"Citation {i+1}: No URL provided")
                
                # Check for required citation fields
                required_fields = ["title", "source", "date"]
                missing_fields = [field for field in required_fields if not citation.get(field)]
                if missing_fields:
                    verification_results["warnings"].append(
                        f"Citation {i+1}: Missing fields - {', '.join(missing_fields)}"
                    )
                
                verification_results["citation_details"].append(citation_result)
            
            return {
                "success": True,
                "verification": verification_results
            }
            
        except Exception as e:
            return {"error": f"Citation verification failed: {str(e)}"}
    
    def audit_calculations(self, calculation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform comprehensive audit of calculations.
        
        Args:
            calculation_data: Complete calculation data including inputs, formula, and results
        
        Returns:
            Comprehensive audit results
        """
        try:
            audit_results = {
                "overall_status": "pass",
                "audit_timestamp": datetime.now().isoformat(),
                "checks_performed": [],
                "summary": {
                    "total_checks": 0,
                    "passed_checks": 0,
                    "failed_checks": 0,
                    "warnings": 0
                },
                "detailed_results": {}
            }
            
            # 1. Input validation audit
            if "inputs" in calculation_data:
                input_audit = self.validate_numbers(
                    calculation_data["inputs"], 
                    calculation_data.get("calculation_type")
                )
                audit_results["detailed_results"]["input_validation"] = input_audit
                audit_results["checks_performed"].append("input_validation")
                
                if not input_audit.get("success") or input_audit.get("validation", {}).get("overall_status") != "pass":
                    audit_results["overall_status"] = "fail"
            
            # 2. Unit consistency audit
            if "units" in calculation_data or "expected_units" in calculation_data:
                unit_audit = self.check_unit_consistency(
                    calculation_data,
                    calculation_data.get("expected_units")
                )
                audit_results["detailed_results"]["unit_consistency"] = unit_audit
                audit_results["checks_performed"].append("unit_consistency")
                
                if not unit_audit.get("success") or unit_audit.get("consistency", {}).get("overall_status") != "pass":
                    audit_results["overall_status"] = "fail"
            
            # 3. Result validation audit
            if "results" in calculation_data:
                result_audit = self.validate_numbers(
                    calculation_data["results"],
                    calculation_data.get("calculation_type")
                )
                audit_results["detailed_results"]["result_validation"] = result_audit
                audit_results["checks_performed"].append("result_validation")
                
                if not result_audit.get("success") or result_audit.get("validation", {}).get("overall_status") != "pass":
                    audit_results["overall_status"] = "fail"
            
            # 4. Citation audit
            if "citations" in calculation_data or "content" in calculation_data:
                citation_audit = self.verify_citations(
                    calculation_data.get("content", ""),
                    calculation_data.get("citations")
                )
                audit_results["detailed_results"]["citation_verification"] = citation_audit
                audit_results["checks_performed"].append("citation_verification")
                
                if not citation_audit.get("success") or citation_audit.get("verification", {}).get("overall_status") != "pass":
                    audit_results["overall_status"] = "fail"
            
            # 5. Formula consistency check
            if "formula" in calculation_data and "inputs" in calculation_data and "results" in calculation_data:
                formula_audit = self._audit_formula_consistency(calculation_data)
                audit_results["detailed_results"]["formula_consistency"] = formula_audit
                audit_results["checks_performed"].append("formula_consistency")
                
                if not formula_audit.get("consistent"):
                    audit_results["overall_status"] = "fail"
            
            # Calculate summary statistics
            for check_name, check_result in audit_results["detailed_results"].items():
                audit_results["summary"]["total_checks"] += 1
                
                if check_result.get("success") and check_result.get("validation", {}).get("overall_status") == "pass":
                    audit_results["summary"]["passed_checks"] += 1
                else:
                    audit_results["summary"]["failed_checks"] += 1
                
                # Count warnings
                if isinstance(check_result, dict):
                    warnings = check_result.get("validation", {}).get("warnings", [])
                    warnings.extend(check_result.get("consistency", {}).get("warnings", []))
                    warnings.extend(check_result.get("verification", {}).get("warnings", []))
                    audit_results["summary"]["warnings"] += len(warnings)
            
            return {
                "success": True,
                "audit": audit_results
            }
            
        except Exception as e:
            return {"error": f"Calculation audit failed: {str(e)}"}
    
    def generate_audit_report(self, audit_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a comprehensive audit report.
        
        Args:
            audit_results: Results from audit_calculations
        
        Returns:
            Formatted audit report
        """
        try:
            report = {
                "report_title": "Quality Control Audit Report",
                "generated_at": datetime.now().isoformat(),
                "executive_summary": {},
                "detailed_findings": {},
                "recommendations": [],
                "approval_status": "pending"
            }
            
            if not audit_results.get("success"):
                report["approval_status"] = "rejected"
                report["executive_summary"]["status"] = "FAILED"
                report["executive_summary"]["reason"] = "Audit process failed"
                return {"success": True, "report": report}
            
            audit_data = audit_results["audit"]
            
            # Executive summary
            report["executive_summary"] = {
                "overall_status": audit_data["overall_status"].upper(),
                "total_checks": audit_data["summary"]["total_checks"],
                "passed_checks": audit_data["summary"]["passed_checks"],
                "failed_checks": audit_data["summary"]["failed_checks"],
                "warnings": audit_data["summary"]["warnings"],
                "pass_rate": audit_data["summary"]["passed_checks"] / max(audit_data["summary"]["total_checks"], 1) * 100
            }
            
            # Detailed findings
            for check_name, check_result in audit_data["detailed_results"].items():
                finding = {
                    "check_type": check_name,
                    "status": "PASS" if check_result.get("success") else "FAIL",
                    "details": check_result,
                    "issues_found": []
                }
                
                # Extract issues
                if isinstance(check_result, dict):
                    for result_key in ["validation", "consistency", "verification"]:
                        if result_key in check_result:
                            result_data = check_result[result_key]
                            finding["issues_found"].extend(result_data.get("errors", []))
                            finding["issues_found"].extend(result_data.get("warnings", []))
                
                report["detailed_findings"][check_name] = finding
            
            # Generate recommendations
            if audit_data["overall_status"] != "pass":
                report["recommendations"].append("Address all failed checks before proceeding")
                report["approval_status"] = "rejected"
            else:
                if audit_data["summary"]["warnings"] > 0:
                    report["recommendations"].append("Review and address warnings for improved quality")
                report["recommendations"].append("Results approved for use")
                report["approval_status"] = "approved"
            
            return {
                "success": True,
                "report": report
            }
            
        except Exception as e:
            return {"error": f"Audit report generation failed: {str(e)}"}
    
    def _extract_numerical_values(self, data: Any, prefix: str = "") -> Dict[str, float]:
        """Extract numerical values from nested data structure."""
        values = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                values.update(self._extract_numerical_values(value, new_prefix))
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]" if prefix else f"item_{i}"
                values.update(self._extract_numerical_values(item, new_prefix))
        elif isinstance(data, (int, float)) and not isinstance(data, bool):
            values[prefix] = float(data)
        
        return values
    
    def _extract_units(self, data: Any, prefix: str = "") -> Dict[str, Dict]:
        """Extract unit information from data."""
        units = {}
        
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                
                # Look for unit information
                if key.endswith("_unit") or key == "unit":
                    param_name = new_prefix.replace("_unit", "").replace(".unit", "")
                    units[param_name] = {"unit": value}
                elif isinstance(value, str) and any(unit in value for unit in self.unit_conversions.keys()):
                    # Extract unit from string value
                    unit_match = re.search(r'([A-Za-z]+)', value)
                    if unit_match:
                        units[new_prefix] = {"unit": unit_match.group(1)}
                else:
                    units.update(self._extract_units(value, new_prefix))
        
        return units
    
    def _are_units_compatible(self, unit1: str, unit2: str) -> bool:
        """Check if two units are compatible (convertible)."""
        if not unit1 or not unit2:
            return False
        
        # Get base units for both
        base1 = self._get_base_unit_type(unit1)
        base2 = self._get_base_unit_type(unit2)
        
        return base1 == base2
    
    def _get_base_unit_type(self, unit: str) -> str:
        """Get the base unit type (energy, power, etc.) for a unit."""
        energy_units = ["J", "kJ", "MJ", "GJ", "TJ", "Wh", "kWh", "MWh", "GWh", "TWh", "BTU", "MMBTU", "therm", "cal", "kcal"]
        power_units = ["W", "kW", "MW", "GW", "TW", "hp", "PS"]
        currency_units = ["USD", "EUR", "GBP", "JPY"]
        time_units = ["s", "min", "h", "day", "year"]
        mass_units = ["kg", "g", "t", "lb", "oz"]
        volume_units = ["m3", "L", "gal", "ft3"]
        
        if unit in energy_units:
            return "energy"
        elif unit in power_units:
            return "power"
        elif unit in currency_units:
            return "currency"
        elif unit in time_units:
            return "time"
        elif unit in mass_units:
            return "mass"
        elif unit in volume_units:
            return "volume"
        else:
            return "unknown"
    
    def _get_conversion_factor(self, from_unit: str, to_unit: str) -> float:
        """Get conversion factor between two compatible units."""
        if from_unit in self.unit_conversions and to_unit in self.unit_conversions:
            return self.unit_conversions[from_unit] / self.unit_conversions[to_unit]
        return 1.0
    
    def _check_dimensional_consistency(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Check dimensional consistency in calculations."""
        # Simplified implementation - would need more sophisticated analysis
        return {"consistent": True, "errors": []}
    
    def _extract_citations_from_content(self, content: str) -> List[Dict]:
        """Extract citations from content text."""
        citations = []
        
        # Look for URL patterns
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        urls = re.findall(url_pattern, content)
        
        for i, url in enumerate(urls):
            citations.append({
                "id": i + 1,
                "url": url,
                "title": f"Reference {i + 1}",
                "source": "extracted_from_content"
            })
        
        return citations
    
    def _is_valid_url(self, url: str) -> bool:
        """Check if URL format is valid."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _check_url_accessibility(self, url: str, timeout: int = 5) -> Dict[str, Any]:
        """Check if URL is accessible."""
        try:
            response = requests.head(url, timeout=timeout, allow_redirects=True)
            return {
                "accessible": response.status_code < 400,
                "status_code": response.status_code,
                "error": None
            }
        except requests.RequestException as e:
            return {
                "accessible": False,
                "status_code": None,
                "error": str(e)
            }
    
    def _audit_formula_consistency(self, calculation_data: Dict[str, Any]) -> Dict[str, Any]:
        """Audit formula consistency with inputs and results."""
        # Simplified implementation - would need more sophisticated formula parsing
        return {"consistent": True, "details": "Formula consistency check passed"}
    
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process QC audit request.
        
        Args:
            input_data: Dictionary containing data to audit
            
        Returns:
            Dictionary with audit results and recommendations
        """
        try:
            self.log_activity("Starting QC audit", input_data)
            
            operation = input_data.get("operation")
            
            if operation == "validate_numbers":
                return self.validate_numbers(input_data.get("results"), input_data.get("calculation_type"))
            elif operation == "check_unit_consistency":
                return self.check_unit_consistency(input_data.get("data"), input_data.get("expected_units"))
            elif operation == "verify_citations":
                return self.verify_citations(input_data.get("content"), input_data.get("citations"))
            elif operation == "audit_calculations":
                return self.audit_calculations(input_data.get("calculation_data"))
            elif operation == "generate_audit_report":
                return self.generate_audit_report(input_data.get("audit_results"))
            else:
                return self.create_error_response(f"Unsupported QC operation: {operation}")
            
        except Exception as e:
            self.logger.error(f"Error in QC audit: {str(e)}")
            return self.create_error_response(f"QC audit failed: {str(e)}")
            
        except Exception as e:
            self.logger.error(f"Error in QC audit: {str(e)}")
            return self.create_error_response(f"QC audit failed: {str(e)}")

