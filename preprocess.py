def de_duplication(df):
    return df.drop_duplicates()

def noise_remover(df):
    df['Interaction content'] = df['Interaction content'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    df['Ticket Summary'] = df['Ticket Summary'].str.lower().str.replace(r'[^\w\s]', '', regex=True)
    return df

def translate_to_en(texts):
    return texts  # Stub for translation logic 
