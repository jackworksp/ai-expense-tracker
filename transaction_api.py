#!/usr/bin/env python3
"""
Transaction Categorization API using Random Forest
Processes bank statements and predicts transaction categories
"""

import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import re
import pickle
import os
import tempfile
from datetime import datetime
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Category mapping based on common transaction patterns
CATEGORY_RULES = {
    'Food & Dining': ['swiggy', 'zomato', 'restaurant', 'cafe', 'kitchen', 'food', 'eat', 'dinner', 'lunch', 'breakfast', 'pizza', 'burger', 'biryani'],
    'Shopping': ['amazon', 'flipkart', 'myntra', 'mall', 'store', 'mart', 'shop', 'retail', 'purchase'],
    'Transportation': ['uber', 'ola', 'petrol', 'diesel', 'fuel', 'parking', 'toll', 'metro', 'bus', 'railway', 'irctc'],
    'Utilities': ['electricity', 'water', 'gas', 'internet', 'broadband', 'mobile', 'phone', 'recharge', 'bill'],
    'Entertainment': ['netflix', 'spotify', 'prime', 'movie', 'cinema', 'pvr', 'inox', 'gaming', 'steam'],
    'Healthcare': ['hospital', 'medical', 'pharmacy', 'doctor', 'clinic', 'medicine', 'health', 'diagnostic'],
    'Education': ['school', 'college', 'university', 'course', 'udemy', 'coursera', 'education', 'tuition'],
    'Investment': ['mutual fund', 'sip', 'investment', 'trading', 'stock', 'shares', 'zerodha', 'groww'],
    'Insurance': ['insurance', 'lic', 'policy', 'premium'],
    'ATM': ['atm', 'withdrawal', 'cash'],
    'Transfer': ['upi', 'imps', 'neft', 'rtgs', 'transfer', 'payment'],
    'Salary': ['salary', 'income', 'credit', 'bonus'],
    'EMI/Loan': ['emi', 'loan', 'mortgage', 'repayment'],
    'Rent': ['rent', 'lease', 'landlord'],
    'Others': []
}

class TransactionProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
        self.label_encoder = LabelEncoder()
        self.model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
        self.is_trained = False
        
    def parse_bank_statement(self, file_path):
        """Parse uploaded HDFC bank statement file (Excel or CSV)"""
        _, extension = os.path.splitext(file_path)
        extension = extension.lower()

        if extension in {'.xls', '.xlsx'}:
            df = pd.read_excel(file_path, header=None)
        elif extension == '.csv':
            df = pd.read_csv(file_path, header=None)
        else:
            raise ValueError("Unsupported file format. Please upload an Excel or CSV statement.")

        # Find the header row (contains 'Date')
        header_idx = None
        for idx, row in df.iterrows():
            if any('date' in str(cell).lower() for cell in row.values):
                header_idx = idx
                break
        
        if header_idx is None:
            raise ValueError("Could not find transaction headers")
        
        # Set proper headers and extract data
        df.columns = df.iloc[header_idx].values
        df = df.iloc[header_idx + 2:]  # Skip header and separator
        
        # Clean column names
        df.columns = ['Date', 'Narration', 'ChqRefNo', 'ValueDate', 'Withdrawal', 'Deposit', 'Balance']
        
        # Filter valid transactions
        df = df.dropna(subset=['Date'])
        df = df[~df['Date'].astype(str).str.contains('\\*')]
        
        # Convert data types
        df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%y', errors='coerce')
        df['Withdrawal'] = pd.to_numeric(df['Withdrawal'], errors='coerce').fillna(0)
        df['Deposit'] = pd.to_numeric(df['Deposit'], errors='coerce').fillna(0)
        df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
        df['Narration'] = df['Narration'].astype(str)
        
        # Add transaction type
        df['Type'] = df.apply(lambda x: 'Debit' if x['Withdrawal'] > 0 else 'Credit', axis=1)
        df['Amount'] = df.apply(lambda x: x['Withdrawal'] if x['Withdrawal'] > 0 else x['Deposit'], axis=1)
        
        return df.dropna(subset=['Date'])
    
    def extract_features(self, transactions):
        """Extract features from transaction data"""
        features = []
        
        for _, row in transactions.iterrows():
            narration = str(row.get('Narration', '')).lower()
            
            # Text features
            text_features = {
                'narration_length': len(narration),
                'word_count': len(narration.split()),
                'has_upi': 1 if 'upi' in narration else 0,
                'has_imps': 1 if 'imps' in narration else 0,
                'has_neft': 1 if 'neft' in narration else 0,
                'has_atm': 1 if 'atm' in narration else 0,
                'is_debit': 1 if str(row.get('Type', '')).lower() == 'debit' else 0,
            }

            # Amount features
            try:
                amount = float(row.get('Amount', 0))
            except (TypeError, ValueError):
                amount = 0.0
            text_features.update({
                'amount': amount,
                'amount_log': np.log1p(amount),
                'is_round_amount': 1 if amount % 100 == 0 else 0,
                'is_small_amount': 1 if amount < 500 else 0,
                'is_large_amount': 1 if amount > 10000 else 0,
            })
            
            # Date features
            date_value = row.get('Date')
            date_features = {
                'day_of_week': np.nan,
                'day_of_month': np.nan,
                'month': np.nan,
                'is_weekend': 0,
                'is_month_end': 0,
            }

            if pd.notna(date_value):
                date_value = pd.to_datetime(date_value, errors='coerce')

            if pd.notna(date_value):
                date_features.update({
                    'day_of_week': date_value.dayofweek,
                    'day_of_month': date_value.day,
                    'month': date_value.month,
                    'is_weekend': 1 if date_value.dayofweek >= 5 else 0,
                    'is_month_end': 1 if date_value.day >= 25 else 0,
                })

            text_features.update(date_features)
            
            features.append(text_features)
        
        return pd.DataFrame(features)
    
    def predict_category_rule_based(self, narration):
        """Rule-based category prediction"""
        narration_lower = str(narration).lower()
        
        for category, keywords in CATEGORY_RULES.items():
            if category == 'Others':
                continue
            for keyword in keywords:
                if keyword in narration_lower:
                    return category
        
        return 'Others'
    
    def prepare_training_data(self, transactions):
        """Prepare training data with rule-based labels"""
        transactions['Category'] = transactions['Narration'].apply(self.predict_category_rule_based)
        
        # Extract features
        features = self.extract_features(transactions)
        
        # Add TF-IDF features
        tfidf_features = self.vectorizer.fit_transform(transactions['Narration'].astype(str))
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Combine features
        X = pd.concat([features.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        
        # Encode labels
        y = self.label_encoder.fit_transform(transactions['Category'])
        
        return X, y, transactions
    
    def train_model(self, transactions):
        """Train Random Forest model"""
        X, y, transactions_with_categories = self.prepare_training_data(transactions)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train model
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Calculate accuracy
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(10)
        
        return {
            'train_accuracy': train_score,
            'test_accuracy': test_score,
            'top_features': feature_importance.to_dict('records'),
            'categories_distribution': pd.Series(self.label_encoder.inverse_transform(y)).value_counts().to_dict()
        }
    
    def predict(self, transactions):
        """Predict categories for new transactions"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Extract features
        features = self.extract_features(transactions)
        
        # Add TF-IDF features
        tfidf_features = self.vectorizer.transform(transactions['Narration'].astype(str))
        tfidf_df = pd.DataFrame(tfidf_features.toarray(), 
                                columns=[f'tfidf_{i}' for i in range(tfidf_features.shape[1])])
        
        # Combine features
        X = pd.concat([features.reset_index(drop=True), tfidf_df.reset_index(drop=True)], axis=1)
        
        # Predict
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        # Get categories
        categories = self.label_encoder.inverse_transform(predictions)
        
        # Get confidence scores
        max_probs = probabilities.max(axis=1)
        
        return categories, max_probs

# Initialize processor
processor = TransactionProcessor()

@app.route('/api/upload', methods=['POST'])
def upload_and_train():
    """Upload bank statement and train model"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        temp_dir = tempfile.mkdtemp(prefix='uploads_')
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)

        # Parse transactions
        transactions = processor.parse_bank_statement(filepath)

        # Train model
        training_results = processor.train_model(transactions)
        
        # Add predictions to transactions
        categories, confidences = processor.predict(transactions)
        transactions['Predicted_Category'] = categories
        transactions['Confidence'] = confidences
        
        # Prepare response
        response = {
            'status': 'success',
            'total_transactions': len(transactions),
            'training_results': training_results,
            'sample_predictions': transactions[['Date', 'Narration', 'Amount', 'Type', 'Predicted_Category', 'Confidence']].head(10).to_dict('records')
        }
        
        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        try:
            if 'filepath' in locals() and os.path.exists(filepath):
                os.remove(filepath)
            if 'temp_dir' in locals() and os.path.isdir(temp_dir):
                os.rmdir(temp_dir)
        except OSError:
            pass

@app.route('/api/predict', methods=['POST'])
def predict_transactions():
    """Predict categories for new transactions"""
    try:
        if not processor.is_trained:
            return jsonify({'error': 'Model not trained. Please upload training data first.'}), 400
        
        data = request.json
        if not data or 'transactions' not in data:
            return jsonify({'error': 'No transactions provided'}), 400

        # Convert to DataFrame
        df = pd.DataFrame(data['transactions'])

        if df.empty:
            return jsonify({'error': 'No transactions provided'}), 400

        # Normalize columns (case-insensitive aliases)
        df.columns = [str(col).strip() for col in df.columns]

        def normalize_header(name):
            return re.sub(r'[^a-z0-9]+', '', str(name).lower())

        normalized_to_original = {}
        for col in df.columns:
            normalized_key = normalize_header(col)
            if normalized_key and normalized_key not in normalized_to_original:
                normalized_to_original[normalized_key] = col

        def resolve_column(*aliases):
            for alias in aliases:
                normalized_alias = normalize_header(alias)
                if normalized_alias in normalized_to_original:
                    return normalized_to_original[normalized_alias], alias
            return None, None

        date_source, _ = resolve_column('date', 'transaction_date', 'value_date')
        if date_source:
            df['Date'] = pd.to_datetime(df[date_source], errors='coerce')
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        else:
            df['Date'] = pd.NaT

        amount_source, amount_alias = resolve_column('amount', 'transaction_amount', 'withdrawal', 'deposit', 'credit', 'debit')
        if amount_source:
            df['Amount'] = pd.to_numeric(df[amount_source], errors='coerce').fillna(0)
        elif 'Amount' in df.columns:
            df['Amount'] = pd.to_numeric(df['Amount'], errors='coerce').fillna(0)
        else:
            df['Amount'] = 0.0

        type_source, type_alias = resolve_column('type', 'transaction_type', 'credit_debit')
        if type_source:
            df['Type'] = df[type_source].fillna('').astype(str)
        elif 'Type' in df.columns:
            df['Type'] = df['Type'].fillna('').astype(str)
        else:
            df['Type'] = np.where(df['Amount'] >= 0, 'Debit', 'Credit')

        # Adjust inferred type if the amount alias implies the direction
        if not type_source and amount_alias in {'deposit', 'credit'}:
            df['Type'] = 'Credit'
        elif not type_source and amount_alias in {'withdrawal', 'debit'}:
            df['Type'] = 'Debit'

        narration_source, narration_alias = resolve_column('narration', 'description', 'details', 'transaction_details')
        if narration_source and narration_source != 'Narration':
            df['Narration'] = df[narration_source].fillna('').astype(str)
        elif 'Narration' in df.columns:
            df['Narration'] = df['Narration'].fillna('').astype(str)
        else:
            df['Narration'] = ''

        # Predict categories
        categories, confidences = processor.predict(df)
        
        # Prepare response
        results = []
        for i, (cat, conf) in enumerate(zip(categories, confidences)):
            results.append({
                'transaction': data['transactions'][i],
                'predicted_category': cat,
                'confidence': float(conf)
            })
        
        return jsonify({'predictions': results})
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get available categories"""
    return jsonify({'categories': list(CATEGORY_RULES.keys())})

@app.route('/api/stats', methods=['POST'])
def get_statistics():
    """Get transaction statistics"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        filepath = os.path.join('/tmp', file.filename)
        file.save(filepath)
        
        # Parse transactions
        transactions = processor.parse_bank_statement(filepath)
        
        # Calculate statistics
        stats = {
            'total_transactions': len(transactions),
            'total_debit': float(transactions['Withdrawal'].sum()),
            'total_credit': float(transactions['Deposit'].sum()),
            'average_transaction': float(transactions['Amount'].mean()),
            'date_range': {
                'start': str(transactions['Date'].min()),
                'end': str(transactions['Date'].max())
            },
            'transaction_types': transactions['Type'].value_counts().to_dict()
        }
        
        # If model is trained, add category stats
        if processor.is_trained:
            categories, _ = processor.predict(transactions)
            transactions['Category'] = categories
            stats['category_distribution'] = transactions.groupby('Category').agg({
                'Amount': ['sum', 'mean', 'count']
            }).to_dict()
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check"""
    return jsonify({
        'status': 'healthy',
        'model_trained': processor.is_trained
    })

if __name__ == '__main__':
    # Train on the provided file initially
    sample_file = '/mnt/user-data/uploads/Acct_Statement_8806_09112025_13_37_48.xls'
    if os.path.exists(sample_file):
        print("Loading and training on sample data...")
        transactions = processor.parse_bank_statement(sample_file)
        training_results = processor.train_model(transactions)
        print(f"Model trained! Accuracy: {training_results['test_accuracy']:.2%}")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
