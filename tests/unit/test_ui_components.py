import pytest
from unittest.mock import Mock, patch, MagicMock
import os
import sys
import json

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from ui.pages import load_id2label, IMAGE_MAP

class TestUIComponents:
    """Test cases for UI components."""
    
    @pytest.mark.unit
    def test_load_id2label_success(self, temp_dir, mock_label_mappings):
        """Test successful loading of id2label mapping."""
        # Create the id2label.json file
        id2label_path = os.path.join(temp_dir, 'id2label.json')
        with open(id2label_path, 'w') as f:
            json.dump(mock_label_mappings['id2label'], f)
        
        with patch('ui.pages.os.path.join', return_value=id2label_path):
            result = load_id2label()
            
            assert result == mock_label_mappings['id2label']
    
    @pytest.mark.unit
    def test_load_id2label_file_not_found(self):
        """Test loading id2label when file doesn't exist."""
        with patch('ui.pages.os.path.join', return_value='/nonexistent/path/id2label.json'):
            with pytest.raises(FileNotFoundError):
                load_id2label()
    
    @pytest.mark.unit
    def test_load_id2label_invalid_json(self, temp_dir):
        """Test loading id2label with invalid JSON."""
        id2label_path = os.path.join(temp_dir, 'id2label.json')
        with open(id2label_path, 'w') as f:
            f.write('invalid json content')
        
        with patch('ui.pages.os.path.join', return_value=id2label_path):
            with pytest.raises(json.JSONDecodeError):
                load_id2label()
    
    @pytest.mark.unit
    def test_image_map_structure(self):
        """Test that IMAGE_MAP has the expected structure."""
        assert isinstance(IMAGE_MAP, dict)
        assert len(IMAGE_MAP) > 0
        
        # Check that all values are strings and end with .jpg
        for key, value in IMAGE_MAP.items():
            assert isinstance(key, str)
            assert isinstance(value, str)
            assert value.endswith('.jpg')
    
    @pytest.mark.unit
    def test_image_map_expected_keys(self):
        """Test that IMAGE_MAP contains expected persona keys."""
        expected_keys = [
            "Tech Enthusiast",
            "Foodie Explorer", 
            "Fitness Buff",
            "Fashion Aficionado",
            "Meme Lord"
        ]
        
        for key in expected_keys:
            assert key in IMAGE_MAP
    
    @pytest.mark.unit
    def test_image_map_paths_exist(self):
        """Test that image paths in IMAGE_MAP exist."""
        for key, path in IMAGE_MAP.items():
            # Check if the path exists relative to the project root
            full_path = os.path.join(os.path.dirname(__file__), '..', '..', path)
            assert os.path.exists(full_path), f"Image path {path} for {key} does not exist"
    
    @pytest.mark.unit
    def test_image_map_no_duplicates(self):
        """Test that IMAGE_MAP has no duplicate values."""
        values = list(IMAGE_MAP.values())
        assert len(values) == len(set(values)), "IMAGE_MAP contains duplicate image paths"
    
    @pytest.mark.unit
    def test_image_map_no_empty_values(self):
        """Test that IMAGE_MAP has no empty keys or values."""
        for key, value in IMAGE_MAP.items():
            assert key.strip() != "", "Empty key found in IMAGE_MAP"
            assert value.strip() != "", f"Empty value found for key '{key}' in IMAGE_MAP" 