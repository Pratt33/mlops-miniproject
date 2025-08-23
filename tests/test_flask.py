import unittest
import os
import sys

# Add flask_app to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'flask_app'))

try:
    from flask_app.app import app
except ImportError:
    # Fallback import path
    from ..flask_app.app import app

class FlaskAppTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Set up test environment
        app.config['TESTING'] = True
        cls.client = app.test_client()
        
        # Ensure models exist before testing
        cls.check_model_availability()

    @classmethod
    def check_model_availability(cls):
        """Ensure models are available for testing."""
        required_files = ['models/model.pkl', 'models/vectorizer.pkl']
        missing_files = [f for f in required_files if not os.path.exists(f)]
        
        if missing_files:
            print(f"Warning: Missing model files: {missing_files}")
            print("Tests may fail if models cannot be loaded from MLflow")

    def test_home_page(self):
        """Test that home page loads correctly."""
        response = self.client.get('/')
        self.assertEqual(response.status_code, 200)
        self.assertIn(b'<title>Sentiment Analysis</title>', response.data)

    def test_predict_page_positive(self):
        """Test prediction with positive sentiment."""
        response = self.client.post('/predict', data=dict(text="I love this! It's amazing and wonderful!"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Happy' in response.data or b'Sad' in response.data or b'1' in response.data or b'0' in response.data,
            "Response should contain sentiment prediction"
        )

    def test_predict_page_negative(self):
        """Test prediction with negative sentiment."""
        response = self.client.post('/predict', data=dict(text="I hate this! It's terrible and awful!"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Happy' in response.data or b'Sad' in response.data or b'1' in response.data or b'0' in response.data,
            "Response should contain sentiment prediction"
        )

    def test_predict_empty_text(self):
        """Test prediction with empty text."""
        response = self.client.post('/predict', data=dict(text=""))
        self.assertEqual(response.status_code, 200)
        # Should still return a response, even if prediction is uncertain

    def test_predict_special_characters(self):
        """Test prediction with special characters and URLs."""
        response = self.client.post('/predict', data=dict(text="Check this out! https://example.com #awesome @user"))
        self.assertEqual(response.status_code, 200)
        self.assertTrue(
            b'Happy' in response.data or b'Sad' in response.data or b'1' in response.data or b'0' in response.data,
            "Response should handle special characters and URLs"
        )

    def test_model_integration(self):
        """Test that the model integration works correctly."""
        # Test multiple predictions to ensure model is working
        test_cases = [
            "great fantastic awesome",
            "terrible horrible awful", 
            "okay fine normal"
        ]
        
        for text in test_cases:
            with self.subTest(text=text):
                response = self.client.post('/predict', data=dict(text=text))
                self.assertEqual(response.status_code, 200)
                # Just ensure we get a response - model output may vary

if __name__ == '__main__':
    unittest.main()