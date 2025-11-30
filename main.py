import os
from dotenv import load_dotenv
from src.pipeline import ClinicalPipeline

# Load environment variables
load_dotenv()

def main():
    # Check for API Key
    if not os.getenv("GOOGLE_API_KEY"):
        print("Error: GOOGLE_API_KEY not found in .env file.")
        print("Please create a .env file with your API key.")
        return

    # Sample Medical Text (Synthetic)
    sample_text = """
    Patient: John Doe (DOB: 05/12/1980)
    Date: 2023-10-25
    Location: General Hospital, Room 302
    
    Dr. Smith: Good morning John. What brings you in today?
    Patient: I've been having a really bad headache for the last 3 days. It's mostly on the left side.
    Dr. Smith: Any nausea or sensitivity to light?
    Patient: Yes, a little bit of nausea, no vomiting though. Light definitely bothers me.
    Dr. Smith: Okay. I see your blood pressure is 140/90, which is a bit high. 
    I'm going to prescribe Sumatriptan 50mg to take at the onset of a headache. 
    I also want you to take Ibuprofen 400mg as needed.
    Let's follow up in 2 weeks.
    """

    pipeline = ClinicalPipeline()
    pipeline.run(sample_text)

if __name__ == "__main__":
    main()
