import pandas as pd
import numpy as np
import json
import re
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from dataclasses import dataclass
from pathlib import Path
import warnings
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ValidationResult:
    """Container for validation results."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    statistics: Dict[str, Any]
    outliers: List[int]
    recommendations: List[str]

@dataclass
class DataSchema:
    """Schema definition for persona dataset."""
    required_columns: List[str]
    column_types: Dict[str, str]
    text_min_length: int = 10
    text_max_length: int = 10000
    label_min_count: int = 5
    max_duplicates_ratio: float = 0.1

class DataValidator:
    """Comprehensive data validation for persona classification datasets."""
    
    def __init__(self, schema: Optional[DataSchema] = None):
        self.schema = schema or self._get_default_schema()
        self.validation_results = []
        
    def _get_default_schema(self) -> DataSchema:
        """Get default schema for persona dataset."""
        return DataSchema(
            required_columns=["bio", "posts", "label"],
            column_types={
                "bio": "string",
                "posts": "string", 
                "label": "string"
            },
            text_min_length=10,
            text_max_length=10000,
            label_min_count=5,
            max_duplicates_ratio=0.1
        )
    
    def validate_dataset(self, df: pd.DataFrame, dataset_name: str = "dataset") -> ValidationResult:
        """Comprehensive dataset validation."""
        logger.info(f"Starting validation for {dataset_name}")
        
        errors = []
        warnings = []
        statistics = {}
        outliers = []
        recommendations = []
        
        # 1. Schema validation
        schema_result = self._validate_schema(df)
        errors.extend(schema_result["errors"])
        warnings.extend(schema_result["warnings"])
        
        if schema_result["errors"]:
            logger.error(f"Schema validation failed for {dataset_name}")
            return ValidationResult(False, errors, warnings, statistics, outliers, recommendations)
        
        # 2. Data quality checks
        quality_result = self._validate_data_quality(df)
        errors.extend(quality_result["errors"])
        warnings.extend(quality_result["warnings"])
        statistics.update(quality_result["statistics"])
        
        # 3. Outlier detection
        outlier_result = self._detect_outliers(df)
        outliers = outlier_result["outliers"]
        warnings.extend(outlier_result["warnings"])
        statistics.update(outlier_result["statistics"])
        
        # 4. Content analysis
        content_result = self._analyze_content(df)
        warnings.extend(content_result["warnings"])
        statistics.update(content_result["statistics"])
        recommendations.extend(content_result["recommendations"])
        
        # 5. Label distribution analysis
        label_result = self._analyze_label_distribution(df)
        warnings.extend(label_result["warnings"])
        statistics.update(label_result["statistics"])
        recommendations.extend(label_result["recommendations"])
        
        is_valid = len(errors) == 0
        
        result = ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            statistics=statistics,
            outliers=outliers,
            recommendations=recommendations
        )
        
        self.validation_results.append((dataset_name, result))
        logger.info(f"Validation completed for {dataset_name}. Valid: {is_valid}")
        
        return result
    
    def _validate_schema(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate dataset schema."""
        errors = []
        warnings = []
        statistics = {}
        
        # Check required columns
        missing_columns = set(self.schema.required_columns) - set(df.columns)
        if missing_columns:
            errors.append(f"Missing required columns: {missing_columns}")
        
        # Check column types
        for col, expected_type in self.schema.column_types.items():
            if col in df.columns:
                if expected_type == "string":
                    if not df[col].dtype == 'object':
                        warnings.append(f"Column '{col}' should be string type, found {df[col].dtype}")
                elif expected_type == "numeric":
                    if not pd.api.types.is_numeric_dtype(df[col]):
                        warnings.append(f"Column '{col}' should be numeric type, found {df[col].dtype}")
        
        statistics["total_rows"] = len(df)
        statistics["total_columns"] = len(df.columns)
        statistics["missing_columns"] = list(missing_columns)
        
        return {"errors": errors, "warnings": warnings, "statistics": statistics}
    
    def _validate_data_quality(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Validate data quality metrics."""
        errors = []
        warnings = []
        statistics = {}
        
        # Check for missing values
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            missing_ratio = total_missing / (len(df) * len(df.columns))
            if missing_ratio > 0.1:
                errors.append(f"Too many missing values: {missing_ratio:.2%}")
            else:
                warnings.append(f"Missing values detected: {total_missing} total")
        
        statistics["missing_values"] = missing_counts.to_dict()
        statistics["missing_ratio"] = missing_ratio if total_missing > 0 else 0
        
        # Check text length constraints
        for text_col in ["bio", "posts"]:
            if text_col in df.columns:
                text_lengths = df[text_col].str.len()
                
                too_short = (text_lengths < self.schema.text_min_length).sum()
                too_long = (text_lengths > self.schema.text_max_length).sum()
                
                if too_short > 0:
                    warnings.append(f"{too_short} {text_col} entries are too short (<{self.schema.text_min_length} chars)")
                
                if too_long > 0:
                    warnings.append(f"{too_long} {text_col} entries are too long (>{self.schema.text_max_length} chars)")
                
                statistics[f"{text_col}_avg_length"] = text_lengths.mean()
                statistics[f"{text_col}_min_length"] = text_lengths.min()
                statistics[f"{text_col}_max_length"] = text_lengths.max()
        
        # Check for duplicates
        duplicates = df.duplicated().sum()
        duplicate_ratio = duplicates / len(df)
        
        if duplicate_ratio > self.schema.max_duplicates_ratio:
            errors.append(f"Too many duplicates: {duplicate_ratio:.2%}")
        elif duplicates > 0:
            warnings.append(f"Found {duplicates} duplicate entries")
        
        statistics["duplicates"] = duplicates
        statistics["duplicate_ratio"] = duplicate_ratio
        
        return {"errors": errors, "warnings": warnings, "statistics": statistics}
    
    def _detect_outliers(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect outliers using multiple methods."""
        warnings = []
        statistics = {}
        outliers = []
        
        # Method 1: Text length outliers
        if "bio" in df.columns and "posts" in df.columns:
            combined_lengths = df["bio"].str.len() + df["posts"].str.len()
            
            # IQR method for text length outliers
            Q1 = combined_lengths.quantile(0.25)
            Q3 = combined_lengths.quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            length_outliers = df[(combined_lengths < lower_bound) | (combined_lengths > upper_bound)].index.tolist()
            outliers.extend(length_outliers)
            
            if length_outliers:
                warnings.append(f"Found {len(length_outliers)} text length outliers")
            
            statistics["text_length_outliers"] = len(length_outliers)
            statistics["text_length_stats"] = {
                "mean": combined_lengths.mean(),
                "std": combined_lengths.std(),
                "min": combined_lengths.min(),
                "max": combined_lengths.max(),
                "Q1": Q1,
                "Q3": Q3
            }
        
        # Method 2: Isolation Forest for multivariate outliers
        if len(df) > 10:  # Need sufficient data for isolation forest
            try:
                # Create features for outlier detection
                features = []
                
                if "bio" in df.columns:
                    features.append(df["bio"].str.len())
                if "posts" in df.columns:
                    features.append(df["posts"].str.len())
                if "label" in df.columns:
                    # Encode labels for numeric analysis
                    label_encoder = {label: idx for idx, label in enumerate(df["label"].unique())}
                    features.append(df["label"].map(label_encoder))
                
                if len(features) >= 2:
                    feature_matrix = np.column_stack(features)
                    
                    # Standardize features
                    scaler = StandardScaler()
                    scaled_features = scaler.fit_transform(feature_matrix)
                    
                    # Detect outliers
                    iso_forest = IsolationForest(contamination=0.1, random_state=42)
                    outlier_labels = iso_forest.fit_predict(scaled_features)
                    
                    iso_outliers = df[outlier_labels == -1].index.tolist()
                    outliers.extend(iso_outliers)
                    
                    if iso_outliers:
                        warnings.append(f"Found {len(iso_outliers)} multivariate outliers using Isolation Forest")
                    
                    statistics["isolation_forest_outliers"] = len(iso_outliers)
            
            except Exception as e:
                warnings.append(f"Isolation Forest outlier detection failed: {str(e)}")
        
        # Remove duplicates from outliers list
        outliers = list(set(outliers))
        statistics["total_outliers"] = len(outliers)
        
        return {"warnings": warnings, "statistics": statistics, "outliers": outliers}
    
    def _analyze_content(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze content quality and patterns."""
        warnings = []
        statistics = {}
        recommendations = []
        
        # Analyze text content patterns
        for text_col in ["bio", "posts"]:
            if text_col in df.columns:
                # Check for empty or whitespace-only entries
                empty_entries = df[text_col].str.strip().eq("").sum()
                if empty_entries > 0:
                    warnings.append(f"Found {empty_entries} empty {text_col} entries")
                
                # Check for repetitive content
                text_samples = df[text_col].dropna().head(100)  # Sample for performance
                unique_ratio = text_samples.nunique() / len(text_samples)
                
                if unique_ratio < 0.8:
                    warnings.append(f"Low text uniqueness in {text_col}: {unique_ratio:.2%}")
                
                # Check for special characters and emojis
                emoji_pattern = re.compile(r'[^\w\s]')
                emoji_counts = df[text_col].str.count(emoji_pattern)
                avg_emojis = emoji_counts.mean()
                
                if avg_emojis > 10:
                    warnings.append(f"High emoji usage in {text_col}: {avg_emojis:.1f} per entry")
                
                statistics[f"{text_col}_unique_ratio"] = unique_ratio
                statistics[f"{text_col}_avg_emojis"] = avg_emojis
                statistics[f"{text_col}_empty_entries"] = empty_entries
        
        # Check for potential data leakage
        if "bio" in df.columns and "posts" in df.columns:
            # Check if bio and posts are too similar
            similarity_scores = []
            for _, row in df.head(50).iterrows():  # Sample for performance
                bio_words = set(row["bio"].lower().split())
                posts_words = set(row["posts"].lower().split())
                if bio_words and posts_words:
                    similarity = len(bio_words.intersection(posts_words)) / len(bio_words.union(posts_words))
                    similarity_scores.append(similarity)
            
            if similarity_scores:
                avg_similarity = np.mean(similarity_scores)
                if avg_similarity > 0.7:
                    warnings.append(f"High similarity between bio and posts: {avg_similarity:.2%}")
                
                statistics["bio_posts_similarity"] = avg_similarity
        
        return {"warnings": warnings, "statistics": statistics, "recommendations": recommendations}
    
    def _analyze_label_distribution(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze label distribution and balance."""
        warnings = []
        statistics = {}
        recommendations = []
        
        if "label" in df.columns:
            label_counts = df["label"].value_counts()
            total_samples = len(df)
            
            # Check for class imbalance
            min_count = label_counts.min()
            max_count = label_counts.max()
            imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
            
            if imbalance_ratio > 5:
                warnings.append(f"Severe class imbalance detected: ratio {imbalance_ratio:.1f}")
                recommendations.append("Consider using class weights or oversampling techniques")
            elif imbalance_ratio > 2:
                warnings.append(f"Moderate class imbalance detected: ratio {imbalance_ratio:.1f}")
                recommendations.append("Consider data augmentation for minority classes")
            
            # Check for classes with too few samples
            under_represented = label_counts[label_counts < self.schema.label_min_count]
            if not under_represented.empty:
                warnings.append(f"Classes with insufficient samples: {under_represented.to_dict()}")
                recommendations.append("Consider removing or combining under-represented classes")
            
            # Calculate diversity metrics
            unique_labels = len(label_counts)
            entropy = -sum((count/total_samples) * np.log2(count/total_samples) for count in label_counts)
            max_entropy = np.log2(unique_labels)
            normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
            
            statistics["label_distribution"] = label_counts.to_dict()
            statistics["class_imbalance_ratio"] = imbalance_ratio
            statistics["unique_labels"] = unique_labels
            statistics["label_entropy"] = entropy
            statistics["normalized_entropy"] = normalized_entropy
            statistics["min_class_count"] = min_count
            statistics["max_class_count"] = max_count
        
        return {"warnings": warnings, "statistics": statistics, "recommendations": recommendations}
    
    def generate_validation_report(self, output_path: Optional[str] = None) -> str:
        """Generate a comprehensive validation report."""
        if not self.validation_results:
            return "No validation results available."
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("DATA VALIDATION REPORT")
        report_lines.append("=" * 60)
        report_lines.append("")
        
        for dataset_name, result in self.validation_results:
            report_lines.append(f"Dataset: {dataset_name}")
            report_lines.append("-" * 40)
            
            # Overall status
            status = "✅ VALID" if result.is_valid else "❌ INVALID"
            report_lines.append(f"Status: {status}")
            report_lines.append("")
            
            # Errors
            if result.errors:
                report_lines.append("❌ ERRORS:")
                for error in result.errors:
                    report_lines.append(f"  • {error}")
                report_lines.append("")
            
            # Warnings
            if result.warnings:
                report_lines.append("⚠️  WARNINGS:")
                for warning in result.warnings:
                    report_lines.append(f"  • {warning}")
                report_lines.append("")
            
            # Statistics
            if result.statistics:
                report_lines.append("📊 STATISTICS:")
                for key, value in result.statistics.items():
                    if isinstance(value, dict):
                        report_lines.append(f"  {key}:")
                        for sub_key, sub_value in value.items():
                            report_lines.append(f"    {sub_key}: {sub_value}")
                    else:
                        report_lines.append(f"  {key}: {value}")
                report_lines.append("")
            
            # Outliers
            if result.outliers:
                report_lines.append(f"🔍 OUTLIERS: {len(result.outliers)} detected")
                report_lines.append("")
            
            # Recommendations
            if result.recommendations:
                report_lines.append("💡 RECOMMENDATIONS:")
                for rec in result.recommendations:
                    report_lines.append(f"  • {rec}")
                report_lines.append("")
            
            report_lines.append("=" * 60)
            report_lines.append("")
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            logger.info(f"Validation report saved to {output_path}")
        
        return report
    
    def create_validation_plots(self, output_dir: str = "validation_plots"):
        """Create visualization plots for validation results."""
        Path(output_dir).mkdir(exist_ok=True)
        
        for dataset_name, result in self.validation_results:
            if not result.statistics:
                continue
            
            # Create subdirectory for this dataset
            dataset_dir = Path(output_dir) / dataset_name
            dataset_dir.mkdir(exist_ok=True)
            
            # 1. Label distribution plot
            if "label_distribution" in result.statistics:
                plt.figure(figsize=(12, 6))
                labels = list(result.statistics["label_distribution"].keys())
                counts = list(result.statistics["label_distribution"].values())
                
                plt.bar(range(len(labels)), counts)
                plt.xlabel("Labels")
                plt.ylabel("Count")
                plt.title(f"Label Distribution - {dataset_name}")
                plt.xticks(range(len(labels)), labels, rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(dataset_dir / "label_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Text length distribution
            if "bio_avg_length" in result.statistics or "posts_avg_length" in result.statistics:
                plt.figure(figsize=(10, 6))
                
                # This would need the original dataframe to create proper histograms
                # For now, just show the statistics
                text_stats = {k: v for k, v in result.statistics.items() if "length" in k}
                if text_stats:
                    plt.text(0.1, 0.5, f"Text Length Statistics:\n{json.dumps(text_stats, indent=2)}", 
                            transform=plt.gca().transAxes, fontsize=10, 
                            verticalalignment='center', fontfamily='monospace')
                    plt.title(f"Text Length Analysis - {dataset_name}")
                    plt.axis('off')
                    plt.savefig(dataset_dir / "text_length_analysis.png", dpi=300, bbox_inches='tight')
                    plt.close()
        
        logger.info(f"Validation plots saved to {output_dir}")

def validate_training_data(main_data_path: str = "persona_dataset.csv", 
                          feedback_data_path: Optional[str] = "result.csv",
                          output_dir: str = "validation_results") -> ValidationResult:
    """Convenience function to validate training data."""
    
    # Create output directory
    Path(output_dir).mkdir(exist_ok=True)
    
    # Initialize validator
    validator = DataValidator()
    
    # Validate main dataset
    try:
        df_main = pd.read_csv(main_data_path)
        main_result = validator.validate_dataset(df_main, "main_dataset")
    except Exception as e:
        logger.error(f"Failed to load main dataset: {e}")
        return ValidationResult(False, [f"Failed to load main dataset: {e}"], [], {}, [], [])
    
    # Validate feedback dataset if it exists
    if feedback_data_path and Path(feedback_data_path).exists():
        try:
            df_feedback = pd.read_csv(feedback_data_path)
            feedback_result = validator.validate_dataset(df_feedback, "feedback_dataset")
        except Exception as e:
            logger.error(f"Failed to load feedback dataset: {e}")
            feedback_result = ValidationResult(False, [f"Failed to load feedback dataset: {e}"], [], {}, [], [])
    else:
        feedback_result = ValidationResult(True, [], ["No feedback dataset found"], {}, [], [])
    
    # Generate report
    report_path = Path(output_dir) / "validation_report.txt"
    report = validator.generate_validation_report(str(report_path))
    
    # Create plots
    validator.create_validation_plots(str(Path(output_dir) / "plots"))
    
    # Return combined result
    combined_errors = main_result.errors + feedback_result.errors
    combined_warnings = main_result.warnings + feedback_result.warnings
    combined_recommendations = main_result.recommendations + feedback_result.recommendations
    
    is_valid = len(combined_errors) == 0
    
    return ValidationResult(
        is_valid=is_valid,
        errors=combined_errors,
        warnings=combined_warnings,
        statistics={
            "main_dataset": main_result.statistics,
            "feedback_dataset": feedback_result.statistics
        },
        outliers=main_result.outliers + feedback_result.outliers,
        recommendations=combined_recommendations
    )

if __name__ == "__main__":
    # Example usage
    result = validate_training_data()
    print(f"Validation completed. Valid: {result.is_valid}")
    if result.errors:
        print("Errors:", result.errors)
    if result.warnings:
        print("Warnings:", result.warnings) 