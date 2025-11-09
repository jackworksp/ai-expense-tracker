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
POST /api/auth/register
POST /api/auth/login
POST /api/transactions/upload
GET  /api/transactions/list
PUT  /api/transactions/confirm
GET  /api/analytics/summary
```

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

