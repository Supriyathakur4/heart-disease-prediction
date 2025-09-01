import requests
import pandas as pd

# Flask API URL
url = "http://127.0.0.1:5000/predict"

# Example patients
patients = [
    {
        "Age": 45,
        "Sex": "M",
        "ChestPainType": "ATA",
        "RestingBP": 120,
        "Cholesterol": 200,
        "FastingBS": 0,
        "RestingECG": "Normal",
        "MaxHR": 150,
        "ExerciseAngina": "N",
        "Oldpeak": 1.5,
        "ST_Slope": "Up"
    },
    {
        "Age": 65,
        "Sex": "F",
        "ChestPainType": "ASY",
        "RestingBP": 160,
        "Cholesterol": 300,
        "FastingBS": 1,
        "RestingECG": "LVH",
        "MaxHR": 120,
        "ExerciseAngina": "Y",
        "Oldpeak": 3.5,
        "ST_Slope": "Flat"
    },
    {
        "Age": 54,
        "Sex": "M",
        "ChestPainType": "NAP",
        "RestingBP": 140,
        "Cholesterol": 250,
        "FastingBS": 0,
        "RestingECG": "ST",
        "MaxHR": 130,
        "ExerciseAngina": "Y",
        "Oldpeak": 2.0,
        "ST_Slope": "Down"
    }
]

# Send request
response = requests.post(url, json={"instances": patients})

# Parse JSON
result = response.json()
print("Raw Response:")
print(result)

# Extract results
if "results" in result:
    results = result["results"]

    # Convert to DataFrame
    df = pd.DataFrame(patients)
    df["Prediction"] = [r["prediction"] for r in results]
    df["Probability"] = [r["probability"] for r in results]
    df["Risk_Level"] = [r["risk_level"] for r in results]
    df["Explanation"] = [r["explanation"] for r in results]

    print("\nFinal Results:")
    print(df)
else:
    print("⚠️ No results returned from API.")


