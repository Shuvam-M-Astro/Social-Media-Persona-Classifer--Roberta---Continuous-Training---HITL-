import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from logic.predict import predict_persona, load_classifier

class TestPredict:
    """Test cases for prediction functionality."""
    
    @pytest.mark.unit
    def test_predict_persona_success(self, mock_classifier):
        """Test successful persona prediction."""
        with patch('logic.predict.classifier', mock_classifier):
            text = "I love coding and technology"
            top_label, scores = predict_persona(text)
            
            assert top_label == "Tech Enthusiast"
            assert isinstance(scores, dict)
            assert len(scores) == 3
            assert scores["Tech Enthusiast"] == 0.8
    
    @pytest.mark.unit
    def test_predict_persona_empty_text(self, mock_classifier):
        """Test prediction with empty text."""
        with patch('logic.predict.classifier', mock_classifier):
            with pytest.raises(ValueError):
                predict_persona("")
    
    @pytest.mark.unit
    def test_predict_persona_none_text(self, mock_classifier):
        """Test prediction with None text."""
        with patch('logic.predict.classifier', mock_classifier):
            with pytest.raises(ValueError):
                predict_persona(None)
    
    @pytest.mark.unit
    def test_predict_persona_classifier_not_initialized(self):
        """Test prediction when classifier is not initialized."""
        with patch('logic.predict.classifier', None):
            with pytest.raises(RuntimeError, match="Classifier not initialized"):
                predict_persona("test text")
    
    @pytest.mark.unit
    def test_predict_persona_classifier_error(self, mock_classifier):
        """Test prediction when classifier raises an error."""
        mock_classifier.side_effect = Exception("Model error")
        
        with patch('logic.predict.classifier', mock_classifier):
            with pytest.raises(Exception, match="Model error"):
                predict_persona("test text")
    
    @pytest.mark.unit
    def test_load_classifier_success(self, temp_dir):
        """Test successful classifier loading."""
        with patch('logic.predict.pipeline') as mock_pipeline:
            mock_pipeline.return_value = Mock()
            
            # Mock the model directory
            with patch('logic.predict.MODEL_DIR', temp_dir):
                load_classifier()
                
                mock_pipeline.assert_called_once_with(
                    "text-classification",
                    model=temp_dir,
                    tokenizer=temp_dir,
                    top_k=None
                )
    
    @pytest.mark.unit
    def test_load_classifier_pipeline_error(self, temp_dir):
        """Test classifier loading when pipeline fails."""
        with patch('logic.predict.pipeline') as mock_pipeline:
            mock_pipeline.side_effect = Exception("Pipeline error")
            
            with patch('logic.predict.MODEL_DIR', temp_dir):
                with pytest.raises(RuntimeError, match="Could not initialize model"):
                    load_classifier()
    
    @pytest.mark.unit
    def test_predict_persona_input_validation(self, mock_classifier):
        """Test input validation for predict_persona."""
        with patch('logic.predict.classifier', mock_classifier):
            # Test with whitespace-only text
            with pytest.raises(ValueError):
                predict_persona("   ")
            
            # Test with very short text
            with pytest.raises(ValueError):
                predict_persona("hi")
    
    @pytest.mark.unit
    def test_predict_persona_score_formatting(self, mock_classifier):
        """Test that scores are properly formatted."""
        with patch('logic.predict.classifier', mock_classifier):
            text = "I love technology"
            top_label, scores = predict_persona(text)
            
            # Check that all scores are between 0 and 1
            for score in scores.values():
                assert 0 <= score <= 1
            
            # Check that scores sum to approximately 1
            assert abs(sum(scores.values()) - 1.0) < 0.01 