# Config.py

class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test (if needed for different label testing)
    TYPE_COLS = ['Type 1', 'Type 2', 'Type 3', 'Type 4']
    
    # Choose the main label column for classification (update this based on your requirement)
    CLASS_COL = 'Type 1'  # Replace with 'Type 2', 'Type 3', or 'Type 4' if needed
    
    # Grouping column if applicable
    GROUPED = 'Innso TYPOLOGY_TICKET'
