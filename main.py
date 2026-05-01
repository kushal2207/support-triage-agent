import os
import pandas as pd
from tqdm import tqdm
from corpus_loader import CorpusLoader
from retriever import Retriever
from agent import SupportAgent

def main():
    """
    Orchestrates the support triage process.
    """
    # Configuration
    data_dir = "data"
    # Looking for 'support_issues.csv' as per instructions, fallback to 'support_tickets.csv'
    input_paths = [
        os.path.join("support_tickets", "support_issues.csv"),
        os.path.join("support_tickets", "support_tickets.csv")
    ]
    input_csv = next((p for p in input_paths if os.path.exists(p)), None)
    output_csv = os.path.join("support_tickets", "output.csv")
    
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        print("CRITICAL ERROR: ANTHROPIC_API_KEY environment variable not set.")
        return

    if not input_csv:
        print(f"CRITICAL ERROR: Input CSV not found in support_tickets/.")
        return

    print("--- Support Triage Agent ---")
    
    # 1. Initialize Corpus
    print(f"Loading corpus from '{data_dir}'...")
    loader = CorpusLoader(data_dir)
    documents = loader.load_corpus()
    print(f"Successfully indexed {len(documents)} documents.")

    # 2. Initialize Retrieval & Agent
    retriever = Retriever(documents)
    agent = SupportAgent(api_key)

    # 3. Read Tickets
    print(f"Reading tickets from '{input_csv}'...")
    df = pd.read_csv(input_csv)
    # Ensure expected columns exist
    required_cols = ['Issue', 'Subject', 'Company']
    for col in required_cols:
        if col not in df.columns:
            print(f"ERROR: Missing required column '{col}' in input CSV.")
            return

    results = []
    
    # 4. Process Loop with Progress Bar
    print(f"Processing {len(df)} tickets...")
    for index, row in tqdm(df.iterrows(), total=len(df), desc="Triaging Tickets"):
        ticket = row.to_dict()
        
        # Keyword-based context retrieval
        query = f"{ticket.get('Subject', '')} {ticket.get('Issue', '')}"
        context_results = retriever.search(query, top_k=3)
        confidence = retriever.get_confidence_score(context_results)
        
        # Agent Classification & Response
        triage_result = agent.process_ticket(ticket, context_results, confidence)
        
        # Merge input data with agent output
        final_record = {**ticket, **triage_result}
        results.append(final_record)

    # 5. Write Output
    print(f"Writing results to '{output_csv}'...")
    output_df = pd.DataFrame(results)
    
    # Enforce output column order as requested
    final_cols = ['Issue', 'Subject', 'Company', 'status', 'product_area', 'response', 'justification', 'request_type']
    # Check if any results were produced
    if not output_df.empty:
        # Reorder and handle any missing columns just in case
        output_df = output_df.reindex(columns=final_cols)
        output_df.to_csv(output_csv, index=False)
        print(f"DONE. {len(output_df)} tickets processed.")
    else:
        print("WARNING: No tickets were processed.")

    print("\nLogs available in: triage_agent.log")

if __name__ == "__main__":
    main()
