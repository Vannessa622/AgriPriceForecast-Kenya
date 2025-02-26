from fastapi import FastAPI,  HTTPException, Query
from pydantic import BaseModel
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import joblib
from typing import List



from fastapi import FastAPI
import logging
import os
from tensorflow.keras.models import load_model

# Load trained model and scaler
model = load_model("lstm_forecasting_model.h5")
scaler = joblib.load("scaler.pkl")  # Load the saved scaler

# Initialize FastAPI app
app = FastAPI()

# Define available commodities and markets
AVAILABLE_COMMODITIES = [
    'Beans (Yellow-Green)', 'Spinach', 'Dry Maize', 'Rice', 'Dry Onions',
    'Finger Millet', 'Cabbages', 'Cowpea leaves (Kunde)', 'Wheat', 'Cowpeas',
    'Beans Red Haricot (Wairimu)', 'Red Irish potato', 'Red Sorghum',
    'Kales/Sukuma Wiki', 'Omena', 'Maize Flour', 'Cow Milk(Processd)',
    'Meat Beef'
]

AVAILABLE_MARKETS = [
    'Aram', 'Kitengela', 'Ngong market', 'Kimumu', 'Makutano Kirinyaga', 'Lomut',
    'Bondo', 'Mwingi Town', 'Ortum', 'Akala', 'Githurai', 'Kangeta', 'Kapcherop',
    'Limuru Rongai Market', 'Kimilili town', 'Bungoma town', 'Chwele', 'Mwatate',
    'Lokichar', 'Wundanyi', 'Kajiado Market', 'Sikhendu', 'Daraja Mbili',
    'Makutano West Pokot', 'Kakamega Town', 'Wangige', 'Siaya', 'Karatina',
    'Naivasha Market', 'Mabera', 'Madaraka', "Wath Ong'er", 'Diani Market',
    'Tseikuru', 'Kawangware', 'Nanyuki Open Air Market', 'Webuye town',
    'Kiminini - Kitale', 'Sibanga', 'Chepareria', 'Amoni', 'Orolwo',
    'Elwak Market', 'Taveta', 'Njoro', 'Kericho Town', 'Langas', 'Kangemi Market',
    'Serem - Nandi', 'Kalundu', 'Serem - Vihiga', 'Kilingili', 'Kipkaren',
    'Kabarnet', 'Marigat', 'Eldama Ravine', 'Mairo-inya', 'Khayega', 'Katito',
    'Kongowea', 'Eldoret Main', 'Othaya', 'Kapsowar', 'Mumias', 'Kutus', 'Ugunja',
    'Sabatia', 'Mbale', 'Nyakoe', 'Nyangusu', 'Kamukuywa', 'Chepseon', 'Holo',
    'Maua', 'Kathonzweni', 'Molo', 'Nkubu', 'Miruka', 'Kagio', 'Kipsitet',
    'Kapkatet', 'Ahero', 'Marindi', 'Keumbu', 'Muhoroni',
    'Cheptiret - Uasin Gishu', 'Nunguni', 'Kerugoya', 'Tawa', 'Mutha', 'Mariakani',
    'Awendo', 'Kibuye', 'Rongo', 'Kasuku', 'Kisasi', 'Embu Town', 'Soko mpya',
    'Mois Bridge', 'Busia Market', 'Sondu- Kericho', 'Luanda', 'Port Victoria',
    'Kabati Kitui', 'Bahati', 'Kamwangi', 'Stadium - Kitale', 'Soko Mjinga Wajir',
    'Kinango', 'Nambale', 'Gatunga', 'Gikomba', 'Cheptulu', 'Majengo Vihiga',
    'Mwangulu', 'Iten', 'Keroka', 'Shika Adabu', 'Kathwana', 'Kapkwen', 'Mulot',
    'Chuka', 'Nyakwere', 'Kebirigo', 'Amukura', 'Ikonge', 'Mukuyu Market',
    'Siakago', 'Oyugis', 'Kakuma Modern Market', 'Engineer', 'Ndhiwa', 'Gongoni',
    'Kangari', 'Kasikeu', 'Lungalunga', 'Mtwapa', 'Chaka', 'Makupa Majengo',
    'Kiritiri', 'Jamhuri Market', 'Makutano Embu', 'Mogotio', 'Lekuru',
    'Runyenjes', 'Muthurwa', 'Suneka', 'Ol-Kalou', 'Kaare', 'Mbita', 'Kambu',
    'Samburu', 'Nyansiongo', 'Kapkayo', 'Ngundune', 'Mpeketoni', 'Emali Market',
    'Nyakongo', "Ololulung'a Market", 'Kabati - Muranga', 'Nginyang',
    'Keroka Kisii', 'Bondeni', 'Nambacha Market', 'Sega', 'Lamu'
]

# Define response models
class MarketResponse(BaseModel):
    markets: List[str]

class CommodityResponse(BaseModel):
    commodities: List[str]

class ValidationResponse(BaseModel):
    message: str

# Endpoints
@app.get("/markets/", response_model=MarketResponse, tags=["Reference Data"])
def get_markets():
    """Returns a list of available markets."""
    return {"markets": sorted(AVAILABLE_MARKETS)}

@app.get("/commodities/", response_model=CommodityResponse, tags=["Reference Data"])
def get_commodities():
    """Returns a list of available commodities."""
    return {"commodities": sorted(AVAILABLE_COMMODITIES)}

# Validation Endpoint: Check if Market & Commodity are Valid
@app.get("/validate/", response_model=ValidationResponse, tags=["Validation"])
def validate_selection(
    market: str = Query(..., description="Market name"),
    commodity: str = Query(..., description="Commodity name")
):
    """Validates if the given market and commodity are available."""
    
    # Check market validity
    if market not in AVAILABLE_MARKETS:
        raise HTTPException(status_code=400, detail=f"Invalid market: {market}. Please choose from available markets.")

    # Check commodity validity
    if commodity not in AVAILABLE_COMMODITIES:
        raise HTTPException(status_code=400, detail=f"Invalid commodity: {commodity}. Please choose from available commodities.")

    return {"message": f"Valid selection: {commodity} in {market}"}


# Request model
class ForecastRequest(BaseModel):
    commodity: str
    market: str
    forecast_days: int

@app.post("/forecast/")
def forecast_prices(request: ForecastRequest):
    """
    API endpoint to forecast future retail prices for a commodity and market.
    """
    commodity, market, forecast_days = request.commodity, request.market, request.forecast_days

    # Load dataset
    data = pd.read_csv("merged_data.csv", parse_dates=["Date"])

    # Filter data for the selected commodity and market
    filtered_data = data[(data["Commodity"] == commodity) & (data["Market"] == market)]
    if filtered_data.empty:
        return {"error": f"No data available for {commodity} in {market}"}

    # Select features used in training
    selected_features = [
        "Retail", "Wholesale_log", "Supply_Volume_log",
        "Retail_Lag1", "Retail_Lag7", "Retail_Rolling3",
        "Wholesale_Rolling3", "Month_sin", "Month_cos"
    ]

    filtered_data = filtered_data[selected_features].dropna()
    data_scaled = scaler.transform(filtered_data)

    # Define sequence length
    seq_length = 14
    current_input = data_scaled[-seq_length:].reshape(1, seq_length, data_scaled.shape[1])

    predictions = []
    for _ in range(forecast_days):
        next_day_scaled = model.predict(current_input, verbose=0)

        # Expand predicted price to match feature count
        next_day_scaled_expanded = np.zeros((1, data_scaled.shape[1]))
        next_day_scaled_expanded[0, 0] = next_day_scaled[0, 0]  # Assign predicted price

        # Inverse transform to original price scale
        next_day_price = scaler.inverse_transform(next_day_scaled_expanded)[0, 0]
        predictions.append(next_day_price)

        # Update input sequence
        next_input = np.vstack((current_input[0, 1:], next_day_scaled_expanded))
        current_input = next_input.reshape(1, seq_length, data_scaled.shape[1])

    # Generate future dates
    future_dates = pd.date_range(start=data["Date"].max() + pd.Timedelta(days=1), periods=forecast_days)
    forecast_result = [{"date": str(future_dates[i]), "predicted_price": predictions[i]} for i in range(forecast_days)]

    return {"commodity": commodity, "market": market, "forecast": forecast_result}


# start api
# uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# url
# http://127.0.0.1:8000/docs
