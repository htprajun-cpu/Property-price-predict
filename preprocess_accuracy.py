import pandas as pd
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter
from tqdm import tqdm

# Update this to your local filename
INPUT_FILE = 'chennai-properties.csv' 
OUTPUT_FILE = 'global_master_geocoded.csv'
ADDRESS_COL = 'location' 

geolocator = Nominatim(user_agent="property_agent_v1")
# Safety Key: 1.1s delay to prevent IP ban
geocode_safe = RateLimiter(geolocator.geocode, min_delay_seconds=1.3)

def run():
    df = pd.read_csv(INPUT_FILE)
    tqdm.pandas(desc="Geocoding Progress")
    
    print(f"Geocoding {len(df)} rows. This will take approx {round(len(df)*1.3/60)} minutes.")
    
    # Run safe geocoding
    df['geo'] = df[ADDRESS_COL].progress_apply(geocode_safe)
    df['lat'] = df['geo'].apply(lambda x: x.latitude if x else None)
    df['lon'] = df['geo'].apply(lambda x: x.longitude if x else None)
    
    df.drop(columns=['geo']).to_csv(OUTPUT_FILE, index=False)
    print(f"✅ Enriched data saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run()