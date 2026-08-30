"""
Generates synthetic benign (label=0) training examples containing realistic PII/contact-detail patterns.
Used to eliminate training-data label bias where contact details/PII patterns only appeared in jailbreak examples.

Tag: source=synthetic_benign_pii
"""
import pandas as pd
from pathlib import Path

BENIGN_PII_EXAMPLES = [
    # Emails - various placements & sentence structures
    "Please send the confirmation email to john.doe@example.com once processed.",
    "My email is jane.smith@example.org. Could you update my profile?",
    "If you have any questions, reach out to support@samplecorp.net directly.",
    "Contact user.test@example.com for further instructions regarding the shipment.",
    "I received a message from admin@testdomain.org asking to verify my account.",
    "Can you forward the meeting notes to alex.brown@example.com?",
    "Please send the report to supervisor@company.internal and cc dev-team@example.com.",
    "You can email client.services@example.com to request an invoice.",
    "My primary contact email address is robert.johnson@example.com.",
    "For inquiries, please drop a note to billing@demo-service.org.",
    "Draft a professional follow-up email to customer.help@example.com.",
    "Write a polite response to info@partner-org.com regarding our project timeline.",
    
    # Phone numbers - various formats & placements
    "You can call me back at 555-123-4567 during standard office hours.",
    "My phone number is 555-987-6543 if you need immediate clarification.",
    "Please reach our customer service desk at (555) 234-5678 for assistance.",
    "His contact number was listed as +1-555-345-6789 on the registration form.",
    "Call 555.456.7890 if there are any updates regarding my appointment.",
    "If I am unreachable, try calling my secondary line at 555-876-5432.",
    "Please text me at 555-654-3210 when the package arrives.",
    "Contact our office receptionist at (555) 789-0123 for scheduling.",
    "Can you check if 555-432-1098 is the correct phone number for customer support?",
    "Please remind me to dial +1 (555) 901-2345 tomorrow morning.",
    
    # SSNs - application/tax/HR contexts
    "My SSN for the background check application is 000-12-3456.",
    "The tax form requires a social security number like 999-12-3456 for filing.",
    "Please verify that the employee SSN listed on file is 000-99-8888.",
    "For identity verification purposes, the reference SSN is 000-33-4455.",
    "Make sure to securely store the applicant SSN 000-55-6677 in the HR portal.",
    
    # Credit card numbers (Luhn-valid placeholder numbers)
    "The payment was processed using card 4111 1111 1111 1111 for the order.",
    "Please record the billing card ending in 4242-4242-4242-4242 for the monthly subscription.",
    "I paid with my Visa card 4000-0000-0000-0000 yesterday.",
    "Confirm the receipt for transaction on card number 4111-2222-3333-4444.",
    "The client updated their card info to 4000 1234 5678 9010 on the account page.",
    
    # Combined PII items (email + phone, name + email + phone, etc.)
    "My email is Sarah.connor@example.com and my phone number is 555-321-7654.",
    "Contact John at john.miller@example.org or call him at (555) 432-8765.",
    "Please update my file: Email: michael.b@example.com, Phone: 555-210-9876.",
    "For urgent queries, email urgency@example.com or dial 555-999-0000.",
    "Application details: Contact: david.w@example.com | Tel: +1-555-888-1122 | SSN: 000-77-8899.",
    "Order confirmation: Customer email user.one@example.com, Phone 555-777-2233, Card 4111 1111 1111 1111.",
    "Send the ticket confirmation to tickets@example.com and text a copy to 555-444-5555.",
    "Reach out to HR at hr.dept@example.org or call 555-333-2211 for employee onboarding.",
    "Please send an invitation to meeting.host@example.com or reach out via 555-666-7777.",
    "The form submitted contains Email: test.user@example.com, Phone: 555-111-2222, SSN: 000-44-5566.",
    "If needed, contact tech-support@example.net or phone 555-222-3333.",
    "My updated details are: email david.smith@example.org and phone 555-888-9999.",
    "Please notify alert-user@example.com and SMS to 555-555-0199 upon job completion.",
    "Write a friendly message to welcome new user alice@example.com at phone 555-123-9999."
]


def generate_df() -> pd.DataFrame:
    records = [
        {"text": prompt, "label": 0, "source": "synthetic_benign_pii"}
        for prompt in BENIGN_PII_EXAMPLES
    ]
    df = pd.DataFrame(records)
    return df


def append_to_train_set(train_csv_path: str = "./data/train_set.csv") -> int:
    new_df = generate_df()
    path = Path(train_csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Target dataset file {train_csv_path} does not exist.")
    
    existing_df = pd.read_csv(path)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df.to_csv(path, index=False)
    print(f"Appended {len(new_df)} synthetic benign PII examples to {train_csv_path}.")
    print(f"New total row count: {len(combined_df)}")
    return len(new_df)


if __name__ == "__main__":
    count = append_to_train_set()
    print(f"Successfully generated and appended {count} synthetic benign PII examples.")
