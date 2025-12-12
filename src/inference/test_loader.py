"""
Quick test script to verify the model loader works.

Run from project root:
    python -m src.inference.test_loader
"""

from pathlib import Path


def test_loader():
    """Test loading a model and making a prediction."""
    from .model_loader import TradingModelLoader
    from ..data.loader import load_all_symbols
    
    # Find the most recent checkpoint
    checkpoints_dir = Path("checkpoints")
    model_dirs = [d for d in checkpoints_dir.iterdir() if d.is_dir() and "patchtst-2" in d.name]
    
    if not model_dirs:
        print("No model checkpoints found in ./checkpoints/")
        print("Run extract_model.py first to create a model checkpoint.")
        return
    
    # Use most recent
    model_dir = sorted(model_dirs)[-1]
    print(f"Loading model from: {model_dir}")
    
    # Load the model
    model = TradingModelLoader.load(model_dir)
    print(f"\nModel loaded successfully!")
    print(model)
    
    # Print feature columns
    print(f"\nFeature columns ({len(model.feature_columns)} total):")
    for i, col in enumerate(model.feature_columns):
        print(f"  {i:3d}. {col}")
    
    # Load some test data
    # Note: load_all_symbols already applies feature engineering and combines symbols
    print(f"\nLoading test data...")
    df = load_all_symbols(days=30)  # Just 30 days for quick test
    print(f"Data shape: {df.shape}")
    print(f"Columns: {list(df.columns)[:10]}...")  # Show first 10 columns
    
    # Check we have enough data
    min_rows = model.get_required_history_length()
    print(f"\nMinimum required rows: {min_rows}")
    print(f"Available rows: {len(df)}")
    
    if len(df) < model.seq_len:
        print(f"ERROR: Not enough data! Need at least {model.seq_len} rows.")
        return
    
    # Make prediction
    print(f"\nMaking prediction...")
    result = model.predict_with_context(df)
    
    print(f"\n{'='*50}")
    print("PREDICTION RESULTS")
    print(f"{'='*50}")
    print(f"Current price:     ${result['current_price']:,.2f}")
    print(f"Predicted price:   ${result['predicted_price']:,.2f}")
    print(f"Predicted change:  ${result['predicted_change']:+,.2f}")
    print(f"Predicted return:  {result['predicted_return']:+.2f}%")
    print(f"Direction:         {result['direction'].upper()}")
    print(f"Confidence:        {result['confidence']:.4f}")
    print(f"{'='*50}")
    
    # Test direction prediction
    direction, confidence = model.predict_direction(df)
    print(f"\nDirection signal: {direction} (confidence: {confidence:.4f})")
    
    print("\n✓ Loader test completed successfully!")


if __name__ == "__main__":
    test_loader()

