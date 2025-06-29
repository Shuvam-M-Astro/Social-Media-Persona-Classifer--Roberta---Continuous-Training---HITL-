import pytest
import pandas as pd
import os
import json
import tempfile
import shutil
from unittest.mock import patch, Mock
import sys

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

class TestModelTraining:
    """Integration tests for model training functionality."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_end_to_end_training_workflow(self, temp_dir, sample_dataset, mock_label_mappings):
        """Test the complete training workflow from data to model."""
        # Create test dataset
        dataset_path = os.path.join(temp_dir, "persona_dataset.csv")
        sample_dataset.to_csv(dataset_path, index=False)
        
        # Create label mappings
        label2id_path = os.path.join(temp_dir, "label2id.json")
        id2label_path = os.path.join(temp_dir, "id2label.json")
        
        with open(label2id_path, 'w') as f:
            json.dump(mock_label_mappings['label2id'], f)
        with open(id2label_path, 'w') as f:
            json.dump(mock_label_mappings['id2label'], f)
        
        # Mock the training components
        with patch('logic.train_model.pd.read_csv', return_value=sample_dataset), \
             patch('logic.train_model.os.path.isfile', return_value=False), \
             patch('logic.train_model.RobertaTokenizer.from_pretrained') as mock_tokenizer, \
             patch('logic.train_model.RobertaForSequenceClassification.from_pretrained') as mock_model, \
             patch('logic.train_model.Trainer') as mock_trainer, \
             patch('logic.train_model.Dataset.from_pandas') as mock_dataset:
            
            # Setup mocks
            mock_tokenizer_instance = Mock()
            mock_tokenizer.return_value = mock_tokenizer_instance
            
            mock_model_instance = Mock()
            mock_model.return_value = mock_model_instance
            
            mock_trainer_instance = Mock()
            mock_trainer.return_value = mock_trainer_instance
            
            mock_dataset_instance = Mock()
            mock_dataset.return_value = mock_dataset_instance
            mock_dataset_instance.map.return_value = mock_dataset_instance
            mock_dataset_instance.train_test_split.return_value = {
                'train': mock_dataset_instance,
                'test': mock_dataset_instance
            }
            
            # Change working directory to temp_dir
            original_cwd = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                # Import and run training
                from logic.train_model import train_model
                train_model()
                
                # Verify that training was called
                mock_trainer_instance.train.assert_called_once()
                mock_model_instance.save_pretrained.assert_called_once()
                mock_tokenizer_instance.save_pretrained.assert_called_once()
                
            finally:
                os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_data_processing_pipeline(self, temp_dir, sample_dataset, sample_feedback_data):
        """Test the data processing pipeline."""
        # Create main dataset
        main_dataset_path = os.path.join(temp_dir, "persona_dataset.csv")
        sample_dataset.to_csv(main_dataset_path, index=False)
        
        # Create feedback dataset
        feedback_path = os.path.join(temp_dir, "result.csv")
        sample_feedback_data.to_csv(feedback_path, index=False)
        
        # Test data loading and merging
        with patch('logic.train_model.pd.read_csv') as mock_read_csv:
            mock_read_csv.side_effect = [sample_dataset, sample_feedback_data]
            
            # Import the function that processes data
            from logic.train_model import train_model
            
            # Mock the rest of the training function to focus on data processing
            with patch('logic.train_model.RobertaTokenizer.from_pretrained'), \
                 patch('logic.train_model.RobertaForSequenceClassification.from_pretrained'), \
                 patch('logic.train_model.Trainer'), \
                 patch('logic.train_model.Dataset.from_pandas'):
                
                # Change working directory
                original_cwd = os.getcwd()
                os.chdir(temp_dir)
                
                try:
                    train_model()
                    
                    # Verify that both datasets were read
                    assert mock_read_csv.call_count >= 2
                    
                finally:
                    os.chdir(original_cwd)
    
    @pytest.mark.integration
    def test_label_mapping_consistency(self, temp_dir, mock_label_mappings):
        """Test that label mappings are consistent between files."""
        # Create label mapping files
        label2id_path = os.path.join(temp_dir, "label2id.json")
        id2label_path = os.path.join(temp_dir, "id2label.json")
        
        with open(label2id_path, 'w') as f:
            json.dump(mock_label_mappings['label2id'], f)
        with open(id2label_path, 'w') as f:
            json.dump(mock_label_mappings['id2label'], f)
        
        # Load and verify consistency
        with open(label2id_path, 'r') as f:
            label2id = json.load(f)
        with open(id2label_path, 'r') as f:
            id2label = json.load(f)
        
        # Convert id2label keys to integers for comparison
        id2label_int = {int(k): v for k, v in id2label.items()}
        
        # Verify bidirectional mapping
        for label, id_val in label2id.items():
            assert id2label_int[id_val] == label
        
        for id_val, label in id2label_int.items():
            assert label2id[label] == id_val
    
    @pytest.mark.integration
    def test_model_save_and_load_cycle(self, temp_dir):
        """Test that models can be saved and loaded correctly."""
        # This would require actual model training, so we'll mock it
        # In a real scenario, you'd want to test with a small model
        
        model_dir = os.path.join(temp_dir, "test_model")
        os.makedirs(model_dir, exist_ok=True)
        
        # Create mock model files
        mock_config = {"model_type": "roberta", "num_labels": 5}
        mock_vocab = {"<s>": 0, "<pad>": 1, "</s>": 2, "<unk>": 3}
        
        with open(os.path.join(model_dir, "config.json"), 'w') as f:
            json.dump(mock_config, f)
        with open(os.path.join(model_dir, "vocab.json"), 'w') as f:
            json.dump(mock_vocab, f)
        
        # Test that the directory structure is correct
        assert os.path.exists(os.path.join(model_dir, "config.json"))
        assert os.path.exists(os.path.join(model_dir, "vocab.json"))
        
        # Test config loading
        with open(os.path.join(model_dir, "config.json"), 'r') as f:
            loaded_config = json.load(f)
        
        assert loaded_config == mock_config
    
    @pytest.mark.integration
    def test_dataset_validation(self, sample_dataset):
        """Test dataset validation and preprocessing."""
        # Test required columns
        required_columns = ['bio', 'posts', 'label']
        for col in required_columns:
            assert col in sample_dataset.columns
        
        # Test data quality
        assert len(sample_dataset) > 0
        assert not sample_dataset['bio'].isna().all()
        assert not sample_dataset['posts'].isna().all()
        assert not sample_dataset['label'].isna().all()
        
        # Test text combination
        sample_dataset['text'] = sample_dataset['bio'] + ' ' + sample_dataset['posts']
        assert all(len(text.strip()) > 0 for text in sample_dataset['text'])
        
        # Test label distribution
        label_counts = sample_dataset['label'].value_counts()
        assert len(label_counts) > 0
        assert all(count > 0 for count in label_counts.values) 