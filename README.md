# AI Expense Tracker

Automatically categorizes bank transactions using machine learning.

## Features
- Upload CSV bank statements.
- Auto-categorize transactions (Food, Transport, Shopping, etc.).
- Learn from user confirmations to improve accuracy over time.
- Track spending patterns through dashboards and analytics.

## Architecture Overview
```
CSV Upload → Parse → Extract Features → ML Model → Predict Category → User Confirms → Update Profile
```

### Components

#### Frontend (React)
- Upload CSV statements.
- Display and review categorized transactions.
- Allow users to confirm or correct predicted categories.
- Present dashboards that summarize spending trends.

#### Backend (Flask/FastAPI)
- Parse uploaded CSV files into structured transactions.
- Perform feature extraction for the machine learning model.
- Serve machine learning predictions for transaction categories.
- Handle user authentication and authorization.

#### Database (PostgreSQL)
- Persist user accounts and authentication details.
- Store transactions and associated metadata.
- Track user spending patterns and category histories.
- Manage supported transaction categories.

#### Machine Learning Model (Random Forest)
- Global model shared by all users.
- Personalizes predictions using per-user profile features.
- Targets 85%+ accuracy on transaction categorization.

## Features Used for Prediction
- Transaction amount.
- Time of day the transaction occurred.
- Day of the week.
- User's historical average spending per category.
- User's transaction frequency.

## Tech Stack
- **Frontend:** React, Chart.js
- **Backend:** Python, Flask, scikit-learn, pandas
- **Database:** PostgreSQL
- **Deployment:** AWS (EC2, RDS, S3)

## API Endpoints
```
POST /api/upload
POST /api/predict
GET  /api/categories
POST /api/stats
GET  /api/health
```

### Testing the API with Postman

1. **Create a collection** named `AI Expense Tracker` and add a `{{base_url}}` environment variable that points to your running Flask server (for example, `http://localhost:5000`).
2. **Upload and train** – add a `POST {{base_url}}/api/upload` request that uses `form-data` with a key named `file` to attach an HDFC bank statement Excel or CSV file. A successful call responds with training metrics and a preview of categorized transactions.
3. **Predict categories** – add a `POST {{base_url}}/api/predict` request with a raw JSON body:
   ```json
   {
     "transactions": [
       {"Date": "2024-05-01", "Narration": "SWIGGY ORDER", "Amount": 450, "Type": "Debit"},
       {"Date": "2024-05-02", "Narration": "Salary credit", "Amount": 65000, "Type": "Credit"}
     ]
   }
   ```
   The response returns the predicted category and confidence for each transaction. This endpoint requires that the model has been trained via `/api/upload` first.
4. **Available categories** – add a `GET {{base_url}}/api/categories` request to retrieve the list of rule-based categories exposed by the service.
5. **Transaction statistics** – add a `POST {{base_url}}/api/stats` request mirroring the upload format (attach a statement as `file`) to receive aggregates such as totals, averages, and optional category distribution when the model is trained.
6. **Health check** – add a `GET {{base_url}}/api/health` request to verify the API status and whether the model has been trained in the current session.

## Why This Approach Works
UPI transactions often contain cryptic descriptors such as "UPI-PAYTM-123456" rather than clear merchant names. By leveraging numerical and behavioral features (amount, time, historical spending patterns) alongside a shared machine learning model enriched with user-specific profiles, the system delivers personalized predictions without the overhead of maintaining separate models per user.

## Getting Started
1. Clone the repository and install dependencies for both the frontend and backend services.
2. Configure the PostgreSQL database connection string for the backend.
3. Train or load the Random Forest model and start the backend API service.
4. Launch the React frontend and begin uploading CSV statements to categorize transactions.

## Future Enhancements
- Incorporate OCR for PDF statement ingestion.
- Add budgeting goals and alerts when approaching category limits.
- Explore model interpretability to explain prediction rationale to users.

