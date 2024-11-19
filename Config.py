class Config:
    # Input Columns
    TICKET_SUMMARY = 'Ticket Summary'
    INTERACTION_CONTENT = 'Interaction content'

    # Type Columns to test (update as needed)
    TYPE_COLS = ['Type 1', 'Type 2', 'Type 3', 'Type 4']  # Ensure consistency based on dataset
    CLASS_COL = 'Type 1'  # Replace with 'Type 2', 'Type 3', or 'Type 4' as required
    
    # Grouping column if applicable
    GROUPED = 'Innso TYPOLOGY_TICKET'  # Ensure this matches your dataset
