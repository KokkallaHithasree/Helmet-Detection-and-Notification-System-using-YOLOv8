import csv
from twilio.rest import Client

def search_and_notify(license_plate_numbers, registered_vehicles_csv):
    try:
        # Load license plate numbers from the file
        with open(license_plate_numbers, 'r') as file:
            plate_numbers = [line.strip() for line in file.readlines()]

        # Load registered vehicles data from the CSV file
        with open(registered_vehicles_csv, newline='') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                if row['License Plate'] in plate_numbers:
                    send_notification(row['Mobile Number'], row['Owner Name'], row['License Plate'], row.get('Preferred Language', 'en'))
    except Exception as e:
        print("An error occurred:", e)

def send_notification(phone_number, owner_name, license_plate, preferred_language, state=None):
    try:
        account_sid = #enter your twilio account_sid
        auth_token = #enter your twilio token
        from_number = #enter your twilio number
        
        client = Client(account_sid, auth_token)

        # Language-specific message templates
        messages = {
            'en': f"Hello {owner_name}, your vehicle with license plate {license_plate} has been detected without a helmet.",
            'te': f"హలో {owner_name}, మీ వాహనం {license_plate} లైసెన్స్ ప్లేట్ బాట్టి నిలబడలేదు.",
            'hi': f"नमस्ते {owner_name}, आपके वाहन {license_plate} को हेलमेट के बिना डिटेक्ट किया गया है।",
            'ta': f"வணக்கம் {owner_name}, உங்கள் வாகனம் {license_plate} ஹெல்மெட் இல்லாது கண்டறியப்பட்டுள்ளது.",
            'kn': f"ನಮಸ್ತೆ {owner_name}, ನಿಮ್ಮ {license_plate} ಗಾಡಿಯನ್ನು ಹೆಲ್ಮೆಟ್ ಇಲ್ಲದೆ ಕಂಡುಹಿಡಿದಿದೆ.",
            'ml': f"നമസ്കാരം {owner_name}, നിങ്ങളുടെ വാഹനം {license_plate} ഹെൽമെറ്റ് ഇല്ലാതെ കണ്ടെത്തിയിരിക്കുന്നു.",
            'bn': f"হ্যালো {owner_name}, আপনার গাড়ির লাইসেন্স প্লেট {license_plate} হেলমেট ছাড়াই চিহ্নিত হয়েছে।",
            'gu': f"નમસ્તે {owner_name}, તમારી વાહન {license_plate} લાયસન્સ પ્લેટ હેલમેટ વગર પ્રતિબંધિત છે.",
            'as': f"নমস্কাৰ {owner_name}, আপোনাৰ গাৰৰ গাৰদী {license_plate} হেলমেট বিহীনভাৱে চিন্হিত কৰা হৈছে।"
        }

        # Select message based on preferred language or default to English
        message_body = messages.get(preferred_language, messages['en'])

        # Send notification
        client.messages.create(
            body=message_body,
            from_=from_number,
            to=phone_number
        )

        print(f"Notification sent to {owner_name} at {phone_number} in {preferred_language}: {message_body}")
    except Exception as e:
        print("An error occurred while sending notification:", e)

if __name__ == "__main__":
    license_plate_numbers = "highest_confidence_plate.txt"
    registered_vehicles_csv = "registered_vehicles1.csv"
    search_and_notify(license_plate_numbers, registered_vehicles_csv)
