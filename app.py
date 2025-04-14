from fastapi import FastAPI
from faker import Faker
import time

app = FastAPI()
fake = Faker()

@app.get("/")
def home():
    return {"message": "API is running! Use /generate_file.php?action=lightning"}

@app.get("/generate_file.php")
def generate_lightning_data(action: str = "lightning"):
    if action != "lightning":
        return {"error": "Invalid action"}

    lightning_data = {
        "lightning_data": {
            "5min_record": [
                {
                    "latitude": str(fake.latitude()),
                    "longitude": str(fake.longitude()),
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "address": fake.address(),
                    "flash_type": fake.random_element(elements=["0", "1"]),
                    "peak_current": f"{fake.random_int(-999999, 999999):08d}",
                    "ic_height": f"{fake.random_int(1, 20):03d}",
                    "number_of_sensors": f"{fake.random_int(1, 5):03d}"
                }
                for _ in range(10)
            ]
        }
    }
    return lightning_data


# run uvicorn app:app --reload
# and then 
# whatever you get paste http://127.0.0.1:8000/generate_file.php?action=lightning
# letss gooo

