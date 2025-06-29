import pytest
from unittest.mock import Mock, patch, mock_open
import os
import csv
import tempfile
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from logic.feedback import save_feedback, retrain_model

class TestFeedback:
    """Test cases for feedback functionality."""
    
    @pytest.mark.unit
    def test_save_feedback_new_file(self, temp_dir):
        """Test saving feedback to a new file."""
        feedback_path = os.path.join(temp_dir, "test_feedback.csv")
        
        result = save_feedback(
            bio="I love technology",
            posts="Check out this new AI tool",
            corrected_label="Tech Enthusiast",
            path=feedback_path
        )
        
        assert result is True
        assert os.path.exists(feedback_path)
        
        # Verify the content
        with open(feedback_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 1
        assert rows[0]['bio'] == "I love technology"
        assert rows[0]['posts'] == "Check out this new AI tool"
        assert rows[0]['label'] == "Tech Enthusiast"
    
    @pytest.mark.unit
    def test_save_feedback_existing_file(self, temp_dir):
        """Test saving feedback to an existing file."""
        feedback_path = os.path.join(temp_dir, "test_feedback.csv")
        
        # Create initial file
        with open(feedback_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=['bio', 'posts', 'label'])
            writer.writeheader()
            writer.writerow({
                'bio': 'Initial bio',
                'posts': 'Initial posts',
                'label': 'Initial label'
            })
        
        # Add new feedback
        result = save_feedback(
            bio="New bio",
            posts="New posts",
            corrected_label="New label",
            path=feedback_path
        )
        
        assert result is True
        
        # Verify both entries exist
        with open(feedback_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert len(rows) == 2
        assert rows[1]['bio'] == "New bio"
        assert rows[1]['posts'] == "New posts"
        assert rows[1]['label'] == "New label"
    
    @pytest.mark.unit
    def test_save_feedback_file_error(self):
        """Test saving feedback when file operations fail."""
        with patch('builtins.open', side_effect=PermissionError("Permission denied")):
            result = save_feedback(
                bio="test bio",
                posts="test posts",
                corrected_label="test label",
                path="/invalid/path/test.csv"
            )
            
            assert result is False
    
    @pytest.mark.unit
    def test_save_feedback_empty_inputs(self, temp_dir):
        """Test saving feedback with empty inputs."""
        feedback_path = os.path.join(temp_dir, "test_feedback.csv")
        
        result = save_feedback(
            bio="",
            posts="",
            corrected_label="",
            path=feedback_path
        )
        
        assert result is True
        assert os.path.exists(feedback_path)
    
    @pytest.mark.unit
    def test_save_feedback_special_characters(self, temp_dir):
        """Test saving feedback with special characters."""
        feedback_path = os.path.join(temp_dir, "test_feedback.csv")
        
        result = save_feedback(
            bio="Bio with émojis 🚀 and special chars: áéíóú",
            posts="Posts with quotes: 'single' and \"double\"",
            corrected_label="Label with spaces and symbols!",
            path=feedback_path
        )
        
        assert result is True
        
        # Verify the content is preserved
        with open(feedback_path, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            
        assert rows[0]['bio'] == "Bio with émojis 🚀 and special chars: áéíóú"
        assert rows[0]['posts'] == "Posts with quotes: 'single' and \"double\""
        assert rows[0]['label'] == "Label with spaces and symbols!"
    
    @pytest.mark.unit
    def test_retrain_model_success(self):
        """Test successful model retraining."""
        mock_result = Mock()
        mock_result.stdout = "Training completed successfully"
        
        with patch('logic.feedback.subprocess.run', return_value=mock_result) as mock_run:
            success, log = retrain_model()
            
            assert success is True
            assert log == "Training completed successfully"
            mock_run.assert_called_once()
    
    @pytest.mark.unit
    def test_retrain_model_failure(self):
        """Test model retraining failure."""
        mock_result = Mock()
        mock_result.stderr = "Training failed: Out of memory"
        
        with patch('logic.feedback.subprocess.run', side_effect=Exception("Training failed")) as mock_run:
            success, log = retrain_model()
            
            assert success is False
            assert "Training failed" in log
            mock_run.assert_called_once()
    
    @pytest.mark.unit
    def test_retrain_model_custom_script_path(self):
        """Test model retraining with custom script path."""
        mock_result = Mock()
        mock_result.stdout = "Training completed"
        
        custom_path = "/custom/path/train.py"
        
        with patch('logic.feedback.subprocess.run', return_value=mock_result) as mock_run:
            success, log = retrain_model(script_path=custom_path)
            
            assert success is True
            mock_run.assert_called_once_with([
                "python", custom_path
            ], capture_output=True, text=True, check=True)
    
    @pytest.mark.unit
    def test_retrain_model_subprocess_error(self):
        """Test model retraining with subprocess error."""
        with patch('logic.feedback.subprocess.run', side_effect=FileNotFoundError("Script not found")):
            success, log = retrain_model()
            
            assert success is False
            assert "Script not found" in log 